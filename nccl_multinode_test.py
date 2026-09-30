"""Exercise the trainer's inter-engine NCCL path over a Ray cluster: one actor per GPU,
StatelessProcessGroup + PyNcclCommunicator (as es_lora_multinode.py does), then an
all-reduce and a broadcast of a 3B-model-sized chunk of weights. Run inside a job
after start_ray_cluster (see submit_nccl_multinode_test.sh)."""
import os, socket, sys, time
import ray, torch

@ray.remote(num_gpus=1)
class Rank:
    def __init__(self, rank):
        self.rank = rank
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
    def info(self):
        return socket.gethostname(), ray.util.get_node_ip_address(), os.environ.get("CUDA_VISIBLE_DEVICES"), \
               os.environ.get("NCCL_NET"), os.environ.get("NCCL_SOCKET_IFNAME")
    def free_port(self):
        s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close(); return p
    def init(self, addr, port, world):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        pg = StatelessProcessGroup.create(host=addr, port=port, rank=self.rank, world_size=world)
        self.comm = PyNcclCommunicator(pg, device=self.device)
        return True
    def allreduce(self):
        x = torch.ones(1024, device=self.device) * (self.rank + 1)
        y = self.comm.all_reduce(x); torch.cuda.synchronize()
        return float(y[0])
    def bcast(self, src, numel):
        t = torch.full((numel,), float(self.rank), device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.time()
        self.comm.broadcast(t, src=src, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
        return time.time() - t0, float(t[0]), float(t[-1])

ray.init(address="auto", include_dashboard=False)
world = int(ray.cluster_resources()["GPU"])
ranks = [Rank.remote(i) for i in range(world)]
for i, inf in enumerate(ray.get([r.info.remote() for r in ranks])):
    print(f"rank {i}: host={inf[0]} ip={inf[1]} CUDA_VISIBLE_DEVICES={inf[2]} NCCL_NET={inf[3]} IFNAME={inf[4]}", flush=True)
addr = ray.get(ranks[0].info.remote())[1]; port = ray.get(ranks[0].free_port.remote())
print(f"master {addr}:{port}, world {world}", flush=True)
t0 = time.time(); ray.get([r.init.remote(addr, port, world) for r in ranks]); print(f"NCCL group up in {time.time()-t0:.1f}s", flush=True)
vals = ray.get([r.allreduce.remote() for r in ranks]); expect = world * (world + 1) / 2
assert all(abs(v - expect) < 1e-3 for v in vals), vals
print(f"all_reduce OK: {vals[0]} == {expect}", flush=True)
numel = 3_100_000_000 // 8   # ~1/8 of Qwen2.5-3B per call, 8 calls ~= one full-model broadcast
tot = 0.0
for _ in range(8):
    res = ray.get([r.bcast.remote(0, numel) for r in ranks])
    assert all(abs(a) < 1e-6 and abs(b) < 1e-6 for _, a, b in res), res   # everyone now holds rank 0's zeros
    tot += max(r[0] for r in res)
gb = 8 * numel * 2 / 1e9
print(f"broadcast OK: {gb:.1f} GB from rank 0 to {world-1} ranks in {tot:.2f}s ({gb/tot:.1f} GB/s)", flush=True)
print("NCCL MULTINODE TEST PASSED")
