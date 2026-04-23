AI Infrastructure & Operations
Complete study guide covering all exam domains — GPU hardware, networking fabrics, storage systems, the NVIDIA software stack, Kubernetes-based AI orchestration, MLOps pipelines, observability, and production security — with real-world examples throughout.

NCA-AIIO Certified
4-Hour Course Summary
Production Examples
H100 / A100 / DGX
InfiniBand · NVLink
KServe · Triton · DCGM
Exam Format
60–70
Multiple choice questions
Duration
90 min
Proctored, online
Passing Score
70%
~49 correct answers
Target Role
Associate
Infra / MLOps / Ops
Exam Domain Weights
22%
AI Infrastructure Fundamentals
20%
Networking & Interconnects
18%
Storage Systems
25%
NVIDIA Software Stack
15%
Operations & MLOps
📌 Exam Strategy
NVIDIA Software Stack carries the highest weight (25%). Prioritize CUDA architecture, NGC, Triton Inference Server, and TensorRT. Networking (InfiniBand, NVLink, RoCE) is the second hardest and worth 20%.
01
AI Infrastructure Fundamentals
Understanding what makes AI infrastructure different from traditional compute — scale, memory bandwidth, interconnects, and power density.

Traditional CPU-based servers are designed for task parallelism with a small number of high-frequency cores. AI workloads — especially training large neural networks — require massive data parallelism: thousands of identical multiply-accumulate operations per second. NVIDIA GPUs solve this with thousands of CUDA cores working in parallel.

Why AI Needs Specialized Infrastructure
🧮
Memory Bandwidth
GPUs like H100 have 3.35 TB/s HBM3 bandwidth vs ~50 GB/s for CPU DDR5. Transformer attention layers are memory-bandwidth-bound, not compute-bound.

⚡
FLOPS per Watt
H100 SXM delivers 3,958 TFLOPS BF16 at 700W. A 64-core CPU delivers ~10 TFLOPS at 350W. GPU wins by 200x for matrix ops.

🔗
High-Speed Interconnects
Moving activations between GPUs during backprop requires NVLink (900 GB/s) or InfiniBand (400 Gb/s). Commodity Ethernet (25 Gb/s) creates a bottleneck.

💧
Cooling & Power
A DGX H100 node draws up to 10.2 kW. Data centers need direct liquid cooling (DLC) rather than traditional air cooling to handle power density.

NVIDIA Hardware Product Lines
Product	Purpose	GPU	Real-World Use Case
DGX H100	Flagship AI training system	8× H100 SXM5 80GB	Pre-training GPT-4-class LLMs, Llama-3 fine-tuning
DGX A100	Previous gen training/inference	8× A100 SXM4 80GB	Stable Diffusion training, BERT at scale
HGX H100	OEM server integration	4× or 8× H100	AWS P5 instance type uses HGX H100
DGX SuperPOD	Multi-node AI supercomputer	Up to 256 DGX nodes	Microsoft Azure Eagle (10,000 H100 nodes)
OVX	Industrial Omniverse rendering	RTX 6000 Ada	Digital twin simulations for factories
EGX	Edge AI deployment	A30/T4	Retail inventory detection, smart cameras
Jetson Orin	Embedded AI at the edge	Ampere GPU + CPU SoC	Autonomous robots, medical imaging devices
Real-World Example
OpenAI GPT-4 training reportedly used roughly 25,000 A100 GPUs running for ~100 days. Each node in the cluster is an HGX A100 server connected via InfiniBand HDR (200 Gb/s) through a fat-tree topology. The storage system feeding training data must sustain hundreds of GB/s of read throughput — which is why NFS alone cannot serve at that scale; Lustre or GPFS (IBM Spectrum Scale) is used instead.
02
GPU Architecture & Hardware
Deep dive into Hopper, Ampere, and Ada Lovelace microarchitectures. Understand SM structure, Tensor Cores, NVLink, and MIG.

Hopper Architecture (H100)
The H100 is built on TSMC 4N process and introduces the Transformer Engine — hardware that automatically switches between FP8 and BF16 per layer to maximize throughput on LLM workloads. The H100 has 132 Streaming Multiprocessors (SMs), each with 128 CUDA cores + 4 Tensor Cores (4th gen).

Spec	H100 SXM5	A100 SXM4	A30
Architecture	Hopper	Ampere	Ampere
CUDA Cores	16,896	6,912	3,584
Tensor Cores	528 (4th gen)	432 (3rd gen)	224 (3rd gen)
HBM Memory	80GB HBM3	80GB HBM2e	24GB HBM2
Memory BW	3.35 TB/s	2.0 TB/s	933 GB/s
BF16 TFLOPS	3,958 (sparse)	1,248 (sparse)	330
NVLink BW	900 GB/s (v4)	600 GB/s (v3)	200 GB/s (v3)
TDP	700W	400W	165W
Transformer Engine — What It Does
Key Concept
The Transformer Engine in H100 uses FP8 precision for matrix multiplications (GEMMs) while maintaining accuracy by dynamically scaling and casting back to higher precision. This doubles throughput vs BF16 on the same hardware. GPT-3 fine-tuning on H100 with FP8 runs 2.4× faster than on A100 BF16.
Multi-Instance GPU (MIG)
MIG allows a single A100 or H100 to be partitioned into up to 7 isolated GPU instances. Each instance has dedicated SM slices, memory bandwidth, and L2 cache — full hardware isolation, not just time-sharing. This matters for multi-tenant inference serving.

BASH
Enabling MIG on H100 — 7 instances of 1g.10gb
# Enable MIG mode (requires GPU reset)
nvidia-smi -i 0 --mig-mode=1
sudo nvidia-smi -pm 1

# Create 7× 1g.10gb instances (1 SM slice, 10GB each)
nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb -C

# List created instances
nvidia-smi -L

# Output:
# GPU 0: NVIDIA H100 80GB HGX (UUID: GPU-abc...)
#   MIG 1g.10gb  Device  0: (UUID: MIG-def...)
#   MIG 1g.10gb  Device  1: (UUID: MIG-ghi...)
#   ... × 7

# Use in Kubernetes via nvidia.com/mig-1g.10gb resource limit
NVLink & NVSwitch
Within a DGX H100, all 8 GPUs are connected via NVLink 4.0 through an NVSwitch chip, providing full all-to-all GPU connectivity at 900 GB/s bidirectional per GPU. This eliminates PCIe bottlenecks during gradient aggregation (AllReduce) in data-parallel training.

DGX H100 — Internal GPU Interconnect Topology
GPU 0
GPU 1
GPU 2
GPU 3
NVSwitch (4× chips)
GPU 4
GPU 5
GPU 6
GPU 7
NVLink 4.0 → 900 GB/s bidirectional per GPU · Full mesh all-to-all
Real-World Example — Why NVSwitch Matters
During Megatron-LM tensor parallelism training of a 70B parameter model, each forward pass requires passing activations between GPUs hundreds of times per second. With PCIe Gen4 (64 GB/s), this creates a severe bottleneck. NVSwitch's 900 GB/s means the gradient exchange completes 14× faster, keeping GPU utilization above 90% (vs 30–40% on PCIe-only systems).
03
Networking for AI
InfiniBand, RDMA over Converged Ethernet (RoCE), ConnectX NICs, fat-tree topology, and the NVIDIA Quantum networking platform.

AI training networks have different requirements from traditional data center networks. The key metric is not just bandwidth but latency + bandwidth × message rate. AllReduce operations in distributed training involve hundreds of small messages per second — which makes kernel bypass (RDMA) critical.

InfiniBand vs RoCE vs Ethernet
Technology	Bandwidth	Latency	Use Case	Example
InfiniBand NDR (400G)	400 Gb/s	<1 µs	Large-scale LLM training clusters	DGX SuperPOD, CoreWeave, Lambda Labs
InfiniBand HDR (200G)	200 Gb/s	<1 µs	Mid-size training clusters	AWS P4d instances (EFA based)
RoCEv2 (100G–400G)	Up to 400 Gb/s	1–5 µs	GPU clusters on existing Ethernet	Azure NDm A100 v4, Google A3
Standard Ethernet	10–100 Gb/s	50–200 µs	Inference serving, management	CPU workloads, monitoring traffic
RDMA — Remote Direct Memory Access
Key Concept
RDMA allows GPU memory on one server to be written directly into GPU memory on another server, bypassing the CPU and OS kernel entirely. This is called GPUDirect RDMA. In NCCL AllReduce over InfiniBand, gradients flow GPU→NIC→Network→NIC→GPU without any CPU involvement, reducing latency to ~1 µs and freeing the CPU for other work.
BASH
Testing InfiniBand / RoCE with NCCL
# Install NCCL tests
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests && make MPI=1 MPI_HOME=/usr/local/mpi CUDA_HOME=/usr/local/cuda

# Run AllReduce benchmark across 2 nodes × 8 GPUs each
mpirun -np 16 --hostfile hostfile \
  -x NCCL_IB_HCA=mlx5_0:1 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_SOCKET_IFNAME=eno1 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

# Expected output on 400G IB NDR:
# busbw = ~350 GB/s at 8GB message size
# Expected output on 100G RoCE:
# busbw = ~85 GB/s (limited by NIC, not switch)
Fat-Tree Topology
Large GPU clusters use a fat-tree (Clos) network topology: leaf switches connect to server NICs, and spine switches connect all leaf switches. This provides full bisection bandwidth — any GPU can communicate with any other GPU at full speed. A DGX SuperPOD with 32 DGX H100 nodes uses 2-tier fat-tree via NVIDIA Quantum-2 InfiniBand switches.

Fat-Tree Topology — GPU Cluster
Spine Switch 1
Spine Switch 2
Spine Switch 3
↕ InfiniBand NDR 400G
Leaf SW 1
Leaf SW 2
Leaf SW 3
Leaf SW 4
↕ ConnectX-7 HCAs (2× per server)
DGX H100
DGX H100
DGX H100
DGX H100
NVIDIA Quantum Networking
→
Quantum-2 (QM9700)
— 64-port NDR 400G InfiniBand switch, used in DGX SuperPOD. Supports in-network computing (SHARP) to offload collective operations onto the switch itself.
→
SHARP (Scalable Hierarchical Aggregation Reduction Protocol)
— performs AllReduce directly inside the IB switch fabric, reducing traffic on uplinks by up to 50% for gradient aggregation.
→
ConnectX-7 HCA
— NVIDIA's 400G InfiniBand/Ethernet dual-port NIC with onboard ASIC for RDMA, GPUDirect, and Smart NIC offload (DPU-lite features).
→
BlueField-3 DPU
— full data-center-on-a-chip; offloads storage (NVMe-oF), networking (OVS, IPsec), and telemetry from the host CPU. Used in NVIDIA's zero-trust infrastructure designs.
04
Storage for AI Workloads
Parallel file systems, object storage, NVMe-oF, and tiered storage architectures that can feed GPU clusters at hundreds of GB/s.

AI workloads create extreme I/O patterns: training data loading (random reads of millions of small files), checkpoint saves (large sequential writes of model weights), and dataset caching (read-heavy, locality-sensitive). No single storage tier solves all of these.

Storage Tiers for AI
Tier	Technology	Bandwidth	Latency	Best For
Hot (L0)	Local NVMe (PCIe 5.0)	12+ GB/s/node	~100 µs	Dataset cache, checkpoint reads during training
Warm (L1)	NVMe-oF over RDMA / AllFlash NAS	100–500 GB/s aggregate	200–500 µs	Active training datasets, model registry
Warm (L2)	Lustre / GPFS / WekaFS	1–10 TB/s aggregate	1–5 ms	Large dataset storage, parallel checkpoint save
Cold (L3)	S3 / GCS / Azure Blob	10–100 GB/s per bucket	10–100 ms	Raw data lake, archived model checkpoints
Parallel File Systems
→
Lustre
— Open-source POSIX parallel FS. Used in 3 of the top 5 supercomputers. Separates metadata (MDS) from object storage (OSS). AWS FSx for Lustre provides managed Lustre linked to S3. Real use: Meta uses Lustre for training dataset serving at exabyte scale.
→
GPFS / IBM Spectrum Scale
— Enterprise parallel FS, supports block/file/object. Used in NVIDIA DGX SuperPOD reference architectures alongside DDN or IBM storage.
→
WekaFS
— Software-defined parallel FS on NVMe. NVIDIA uses WekaFS in DGX BasePOD reference design. Delivers 500+ GB/s with low latency via NVMe-oF.
→
VAST Data
— All-flash universal storage that combines NFS/S3/NVMe-oF. Used by several AI hyperscalers. Supports tiering hot data to QLC NVMe and cold data to cloud object storage.
GPUDirect Storage
Key Concept
GPUDirect Storage (GDS) allows data to flow directly from storage (NVMe SSD or NVMe-oF) into GPU memory, bypassing the CPU and system DRAM entirely. Without GDS: Storage → PCIe → CPU DRAM → PCIe → GPU. With GDS: Storage → PCIe → GPU (direct peer-to-peer DMA). This reduces DL data loading CPU utilization by ~60% and increases throughput by 2–4×.
PYTHON
Using GPUDirect Storage via cuFile in PyTorch DataLoader
import cufile
import torch
import numpy as np

# Open file using cuFile driver (bypasses page cache)
with cufile.CuFile("/mnt/nvme/dataset/shard_0001.bin", "r") as f:
    # Allocate directly on GPU
    gpu_buf = torch.empty(1024 * 1024 * 128, dtype=torch.float32, device="cuda:0")
    
    # Read directly from NVMe to GPU memory — no CPU copy
    bytes_read = f.read(gpu_buf, size=gpu_buf.nbytes(), file_offset=0)
    
    # tensor is ready for model.forward() immediately
    logits = model(gpu_buf.reshape(-1, 768))

# Real-world benefit: 3.1× faster ImageNet training iteration
# on a system with 8× Samsung PM9A3 NVMe + 8× A100
Real-World Example — Checkpoint at Scale
Training Llama 2 70B generates 140GB checkpoints (FP32 weights). Saved every 500 steps, that's 140GB × (training_steps / 500) of checkpoint data. Slow checkpoint saves pause training. NVIDIA uses asynchronous checkpoint via PyTorch's torch.save with CPU-offload — while the GPU continues training the next batch, the previous checkpoint streams from GPU→CPU DRAM→Lustre in the background.
05
NVIDIA AI Software Stack
CUDA, cuDNN, TensorRT, NCCL, Triton Inference Server, NGC, and the full software stack that underpins production AI.

NVIDIA AI Software Stack — Layer Model
Application (PyTorch / TF / JAX)
ML Frameworks (cuDNN / cuBLAS / NCCL)
TensorRT / Triton / DeepSpeed
CUDA Runtime / cuFile / NVML
GPU Driver (kernel mode)
GPU Hardware (H100 / A100)
CUDA Architecture
CUDA (Compute Unified Device Architecture) is the programming model and runtime for NVIDIA GPUs. Key concepts for the exam:

Concept	Definition	Exam-Relevant Detail
Grid	Top-level execution domain for a kernel	Grid = collection of thread blocks
Block	Group of threads sharing shared memory	Max 1024 threads/block on H100
Warp	32 threads executing in lock-step (SIMT)	Warp divergence causes serial execution
SM (Streaming Multiprocessor)	Physical CUDA core grouping	H100 has 132 SMs; each runs 64 warps
Shared Memory	Fast on-chip SRAM per block	228KB per SM on H100; programmer-managed
Global Memory	HBM (off-chip DRAM)	80GB on H100 SXM5; bandwidth = 3.35 TB/s
CUDA Stream	Sequence of async operations	Multiple streams = overlapping compute + copy
cuDNN — Deep Neural Network Library
cuDNN provides hardware-accelerated primitives for neural networks: convolutions, normalizations, pooling, activation functions, and attention. PyTorch and TensorFlow call cuDNN automatically. Key: cuDNN chooses the fastest convolution algorithm per hardware at runtime using autotuning (torch.backends.cudnn.benchmark = True).

TensorRT — Inference Optimization
TensorRT takes a trained model and produces an optimized inference engine. It performs: layer fusion (merging Conv+BN+ReLU into one kernel), kernel auto-tuning, INT8/FP8 quantization, and dynamic shape handling.

PYTHON
Converting ONNX model to TensorRT engine (INT8 quantization)
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, calibrator):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)   # Enable INT8
    config.int8_calibrator = calibrator       # Calibration dataset
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 4 << 30  # 4GB workspace
    )

    engine = builder.build_serialized_network(network, config)
    return engine

# Result: ResNet-50 latency drops from 2.1ms (FP32) → 0.4ms (INT8)
# on T4 GPU — 5.3× speedup with <0.5% accuracy loss
NCCL — Collective Communications
NCCL (NVIDIA Collective Communications Library) implements distributed collective operations: AllReduce, AllGather, ReduceScatter, Broadcast, Reduce. These are the building blocks of data-parallel and model-parallel training. NCCL automatically selects the fastest path: NVLink (intra-node) or InfiniBand/RoCE (inter-node).

Triton Inference Server
Triton is NVIDIA's open-source production inference serving system. It supports multiple model frameworks (TensorRT, ONNX, PyTorch, TensorFlow, Python, FIL for XGBoost) in a single server, with dynamic batching, model pipelines, and multi-instance GPU serving.

YAML
Triton model config — LLM with dynamic batching + 2 GPU instances
# /models/llama3-8b/config.pbtxt
name: "llama3-8b"
backend: "vllm"          # vLLM backend for LLM serving
max_batch_size: 0        # 0 = let dynamic batcher decide

model_transaction_policy {
  decoupled: True         # streaming tokens
}

instance_group [
  {
    count: 2               # 2 model instances
    kind: KIND_GPU
    gpus: [ 0, 1 ]          # pinned to GPU 0 and GPU 1
  }
]

parameters {
  key: "max_num_seqs"
  value: { string_value: "256" }
}

dynamic_batching {
  max_queue_delay_microseconds: 500
}
NGC — NVIDIA GPU Cloud
→
NGC Catalog
— Container registry with pre-built, optimized containers for PyTorch, TensorFlow, RAPIDS, Triton, NeMo. Images are validated on DGX hardware.
→
NGC Models
— Pre-trained model weights (Stable Diffusion, LLaMA variants, BERT, ResNet). Download and fine-tune without training from scratch.
→
NeMo Framework
— NVIDIA's end-to-end platform for building, fine-tuning, and deploying LLMs. Supports PEFT (LoRA, P-Tuning), parallelism strategies, and direct export to TensorRT-LLM.
→
TensorRT-LLM
— Optimized inference library for LLMs. Implements PagedAttention, continuous batching, multi-GPU tensor parallelism. Delivers up to 4× throughput vs vanilla HuggingFace transformers on the same GPU.
06
Containerization & Orchestration
NVIDIA Container Toolkit, GPU scheduling in Kubernetes, MIG in K8s, and running GPU workloads with Helm and Karpenter.

NVIDIA Container Toolkit (nvidia-docker)
The Container Toolkit exposes GPU devices to Docker/containerd containers without privileged mode. It injects the CUDA runtime libraries into the container at run time — the container image itself only needs the CUDA headers; the driver stays on the host. This allows one driver version to serve many different CUDA toolkit versions.

BASH
Installing NVIDIA Container Toolkit + running a GPU container
# Install on Ubuntu 22.04
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd  # for K8s
sudo systemctl restart containerd

# Run a GPU container
docker run --runtime=nvidia --gpus all \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  python -c "import torch; print(torch.cuda.get_device_name(0))"
# Output: NVIDIA H100 80GB HGX
NVIDIA GPU Operator for Kubernetes
The GPU Operator automates the installation of: GPU driver (DaemonSet), NVIDIA Container Toolkit, Device Plugin (nvidia.com/gpu resource), DCGM Exporter (metrics), MIG Manager, and Node Feature Discovery (NFD). It replaces manual per-node driver installation in large clusters.

BASH
Installing GPU Operator on EKS via Helm
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \       # Driver pre-installed on AMI
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set mig.strategy=mixed        # supports mixed MIG profiles

# Verify GPUs visible to K8s scheduler
kubectl get nodes -o json | jq '.items[].status.capacity | select(."nvidia.com/gpu")'
# {"nvidia.com/gpu": "8"}
GPU Pod Scheduling
YAML
GPU training job — requesting 4× A100 GPUs with topology hint
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-finetune-job
spec:
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      containers:
      - name: trainer
        image: nvcr.io/nvidia/nemo:24.01
        command: ["torchrun", "--nproc_per_node=4", "train_llm.py"]
        resources:
          limits:
            nvidia.com/gpu: "4"
            memory: "320Gi"
            cpu: "32"
        env:
        - name: NCCL_IB_HCA
          value: "mlx5_0,mlx5_1"
        - name: NCCL_NET_GDR_LEVEL
          value: "5"           # Enable GPUDirect RDMA
      restartPolicy: OnFailure
Karpenter for GPU Node Provisioning
Real-World Example
A machine learning platform team at a fintech uses Karpenter with a NodePool targeting p4d.24xlarge (8× A100) and p3.16xlarge (8× V100) instances. Training jobs are submitted as Argo Workflow DAGs. Karpenter provisions GPU nodes in ~2 min when pods are pending, and terminates them within 5 min of job completion — reducing GPU spend by ~65% vs always-on reserved instances.
YAML
Karpenter NodePool for GPU training workloads (EKS)
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training
spec:
  template:
    metadata:
      labels:
        node-type: gpu-training
    spec:
      requirements:
      - key: karpenter.k8s.aws/instance-family
        operator: In
        values: ["p4d", "p4de"]
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]
      nodeClassRef:
        name: gpu-nodeclass
  limits:
    nvidia.com/gpu: "256"   # max 32 × p4d nodes
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 5m
07
MLOps & Model Deployment
End-to-end MLOps pipelines, model serving patterns, inference optimization, and LLMOps at production scale.

Production MLOps Architecture
End-to-End MLOps Pipeline — NVIDIA Stack
Data Lake (S3)
→
Feature Store (Feast)
→
Training (NeMo / PyTorchJob)
Experiment Tracking (MLflow)
→
Model Registry
→
TensorRT Optimization
Triton Inference Server
→
KServe / Seldon
→
API Gateway
DCGM Exporter → Prometheus → Grafana | Drift detection → Retraining trigger
LLM Inference Optimization Techniques
Technique	What It Does	Benefit	Tool
Continuous Batching	Adds new requests mid-batch, removes finished sequences without waiting	3–10× throughput vs static batching	vLLM, TRT-LLM
PagedAttention	KV-cache stored in non-contiguous pages (like OS virtual memory)	Near-zero KV-cache waste; 24× more concurrent users	vLLM
Speculative Decoding	Small draft model proposes tokens; large model verifies in parallel	2–3× lower latency for autoregressive generation	TRT-LLM, HF
FP8 Quantization	Weights and activations in FP8 instead of FP16/BF16	2× throughput, 50% VRAM reduction	TRT-LLM, bitsandbytes
Tensor Parallelism	Split individual layer weight matrices across GPUs	Fit larger models; reduce per-GPU memory	Megatron-LM, vLLM
Flash Attention 2	Tiled attention computation to avoid HBM round-trips	2–4× faster attention, 10× less memory	flash-attn library
KServe — Kubernetes-Native Model Serving
YAML
KServe InferenceService — Llama 3 8B with Triton backend + canary rollout
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-8b
  namespace: production
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
spec:
  predictor:
    canaryTrafficPercent: 10   # 10% traffic to new version
    triton:
      storageUri: "s3://model-registry/llama3-8b-trtllm/v2"
      runtimeVersion: "24.01-trtllm-python-py3"
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: "40Gi"
    scaleTarget: 30             # target 30 concurrent requests
    scaleMetric: "concurrency"  # autoscale on request concurrency
    minReplicas: 1
    maxReplicas: 8
KEDA — GPU-Aware Autoscaling
Real-World Example
An LLM API provider uses KEDA with a custom DCGM scaler: when average GPU utilization across Triton replicas exceeds 80%, KEDA scales up by 2 pods. When a SQS queue (inference request queue) depth exceeds 100 messages, KEDA scales up immediately. This gives both proactive and reactive scaling without over-provisioning at night (min 1 replica, serving from a single MIG instance).
08
Monitoring & Observability
DCGM, nvidia-smi, Prometheus, Grafana, and production GPU health management.

DCGM — Data Center GPU Manager
DCGM is NVIDIA's tool for GPU health monitoring, diagnostics, and policy management in multi-GPU environments. It exposes ~200 GPU metrics via Prometheus. Critical for production SRE work.

DCGM Metric	Description	Alert Threshold (Typical)
DCGM_FI_DEV_SM_CLOCK	Current SM clock frequency (MHz)	Alert if <80% of base clock
DCGM_FI_DEV_GPU_UTIL	GPU compute utilization (%)	Warning <60% during active training
DCGM_FI_DEV_FB_USED	Framebuffer (GPU VRAM) used (MB)	Alert if >95% for OOM prevention
DCGM_FI_DEV_GPU_TEMP	GPU die temperature (°C)	Warning at 80°C, Critical at 90°C
DCGM_FI_DEV_POWER_USAGE	Current power draw (W)	Alert if >TDP (throttling risk)
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL	Single-bit ECC errors (correctable)	Alert if increasing over time
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL	Double-bit ECC errors (fatal)	Immediate alert — GPU needs replacement
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL	NVLink aggregate bandwidth	Alert if <50% of theoretical max
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE	Tensor core utilization (%)	Low value → model isn't GPU-bound
BASH
Running DCGM diagnostics + watching live metrics
# Install dcgmi CLI
apt-get install -y datacenter-gpu-manager

# Run Level 1 diagnostic (30s) — checks clocks, memory, PCIe
dcgmi diag -r 1

# Run Level 3 diagnostic (90 min) — full stress test
dcgmi diag -r 3

# Watch GPU utilization live
dcgmi dmon -e 1004,1005,1009,1010,155
# #Entity  GPUTL  MCUTL  SMACT  SMOCC  POWER
# GPU 0     87     34     89     91    675.2

# Check NVLink health
dcgmi nvlink --link-status -g 0

# Detect XID errors (GPU crash logs)
nvidia-smi --query-gpu=timestamp,name,xid.last_xid_errors --format=csv
Prometheus + Grafana Stack
YAML
Prometheus alert rules for GPU health
groups:
- name: gpu-health
  rules:
  - alert: GPUHighTemperature
    expr: DCGM_FI_DEV_GPU_TEMP > 85
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "GPU {{ $labels.gpu }} temp {{ $value }}°C"

  - alert: GPUDoubleBitECCError
    expr: increase(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Fatal ECC error on GPU {{ $labels.gpu }} — cordon node immediately"

  - alert: GPULowTensorCoreUtilization
    expr: DCGM_FI_PROF_PIPE_TENSOR_ACTIVE < 0.20 and DCGM_FI_DEV_GPU_UTIL > 50
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "GPU busy but tensor cores idle — check model precision/batch size"
XID Error Codes (Critical for Exam)
XID Code	Meaning	Action
XID 13	Graphics Engine Exception (driver bug or OOM)	Check VRAM usage, restart workload
XID 31	GPU memory page fault	Review pointer arithmetic in CUDA kernel
XID 48	DBE (Double-Bit ECC) error — hardware fault	Cordon node, RMA the GPU
XID 61/79	NVLink error	Check NVLink cable/connector, run diagnostics
XID 92	High Single Bit Error rate — GPU degrading	Schedule maintenance, monitor closely
09
Security & Compliance
Secure boot, NVIDIA Confidential Computing, container image security, IRSA, and zero-trust AI infrastructure.

NVIDIA Confidential Computing (H100)
Key Feature
H100 introduces Confidential Computing — the GPU memory is encrypted with a hardware-level key stored in the GPU itself. Even the hypervisor or cloud provider cannot read the data during computation. This enables privacy-preserving AI for healthcare and financial workloads. AMD's equivalent is SEV-SNP for CPUs; NVIDIA extends it to GPU memory via TEE (Trusted Execution Environment).
Container Security for AI Workloads
→
NGC Image Signing
— All NGC containers are signed with NVIDIA's PGP key and verifiable via cosign/Sigstore. Deploy with
policy.json
in containerd to reject unsigned images.
→
Non-Root Containers
— NVIDIA Container Toolkit supports running as non-root with
--user
flag. GPU device access is managed via device plugin, not privileged mode.
→
PodSecurityAdmission
— Use
restricted
profile in K8s to prevent GPU pods from running privileged, mounting host paths, or using host network — critical for multi-tenant GPU clusters.
→
IRSA (IAM Roles for Service Accounts)
— AWS best practice for GPU training pods on EKS to access S3 training data, ECR images, and Secrets Manager — without storing AWS credentials in pods.
YAML
Secure GPU pod — non-root, read-only root fs, IRSA
apiVersion: v1
kind: Pod
metadata:
  name: secure-inference-pod
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/model-inference-role
spec:
  serviceAccountName: inference-sa   # bound to IAM role via IRSA
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: triton
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
    resources:
      limits:
        nvidia.com/gpu: "1"
BlueField DPU — Zero Trust Networking
NVIDIA BlueField-3 DPUs implement zero-trust networking by running OVS (Open vSwitch) and IPsec on the DPU ARM cores rather than the host CPU. Every East-West packet between GPU nodes is encrypted in hardware with WireGuard/IPsec tunnels established by the DPU — the host OS never sees unencrypted packets. This is used in NVIDIA's AI Enterprise secure infrastructure blueprints.

10
Exam Tips & Quick-Reference Cheatsheet
High-frequency exam topics, memory anchors, and common distractors to avoid.

High-Frequency Topics (Appear in 40%+ of Questions)
★
H100 Transformer Engine
— FP8 precision, automatic precision scaling, 2× throughput vs A100 BF16. Will have questions about what FP8 means and when it helps.
★
MIG partitioning
— 7 instances max on H100/A100, hardware isolation, Kubernetes
nvidia.com/mig-Xg.Ygb
resource syntax.
★
NVLink vs PCIe vs InfiniBand
— NVLink is intra-node GPU-to-GPU; InfiniBand is inter-node; PCIe is CPU-to-GPU. Know the bandwidth numbers.
★
DCGM metrics
— Know what DBE ECC means (GPU replacement) vs SBE (correctable, monitor). XID 48 = fatal hardware error.
★
TensorRT vs Triton
— TensorRT optimizes a model (compile-time). Triton serves models at runtime. They work together: TensorRT engine is loaded by Triton.
★
NCCL AllReduce
— The operation used for gradient synchronization in data-parallel training. Uses NVLink intra-node, InfiniBand inter-node.
★
GPUDirect RDMA / Storage
— Bypasses CPU for GPU-to-GPU data and GPU-to-NVMe data transfer respectively.
★
NGC
— NVIDIA's container/model registry. Know that NGC containers bundle the CUDA libraries but NOT the GPU driver (driver stays on host).
Common Exam Distractors
❌ Wrong Answers to Avoid
"InfiniBand is only for HPC, not AI" — FALSE. InfiniBand is widely used for AI training.
"MIG instances share memory bandwidth" — FALSE. MIG provides hard memory partition isolation.
"TensorRT can be used for training" — FALSE. TensorRT is inference-only.
"nvidia-docker is required for Kubernetes GPU" — FALSE. GPU Operator / NVIDIA Container Toolkit + containerd.
"NCCL only works over InfiniBand" — FALSE. NCCL works over NVLink, IB, RoCE, and TCP.
✓ Correct Distinctions
DGX = NVIDIA's own server (validated, support included). HGX = OEM board for server vendors (Dell, HPE, etc.).
Hopper = H100 architecture. Ampere = A100/A30/A10. Ada = RTX 40-series workstation.
cuDNN = training + inference primitives. TensorRT = inference optimization (model compilation).
DCGM = fleet management / Prometheus metrics. nvidia-smi = single-node CLI monitoring.
ECC DBE = fatal, replace GPU. ECC SBE = correctable, monitor.
Quick-Reference Numbers
Fact	Value
H100 HBM3 bandwidth	3.35 TB/s
H100 NVLink 4.0 bandwidth (per GPU)	900 GB/s bidirectional
Max MIG instances on H100	7× 1g.10gb
H100 BF16 TFLOPS (sparse)	3,958 TFLOPS
InfiniBand NDR bandwidth	400 Gb/s (50 GB/s)
DGX H100 total TDP	10.2 kW
CUDA warp size	32 threads
H100 max threads per block	1024
H100 Streaming Multiprocessors	132 SMs
H100 CUDA cores total	16,896
Final Study Strategy
Week 1: Focus on GPU architecture (Sec 2) + NVIDIA Software Stack (Sec 5) — covers 47% of exam weight. Week 2: Networking (Sec 3) + Storage (Sec 4). Week 3: Practice questions + review Sec 8 monitoring metrics (XID codes, DCGM fields). The exam is scenario-based — for each question, map it to: what hardware is involved → what software layer → what the symptom/metric tells you.
NCA AIIO Study Guide · Author: Ram
Built for production-level exam preparation · All domains covered · ramcreddy-ch
