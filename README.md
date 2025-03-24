# WorldModel_VLA

<br>

[Diffusion](#Diffusion) | [VLA](#VLA) | [Others](#Others) 

<br>


## WorldModel

#### <summary>Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability
Authors: Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, Hongyang Li
<details span>
<summary><b>Abstract</b></summary>
World models can foresee the outcomes of different actions, which is of paramount importance for autonomous driving. Nevertheless, existing driving world models still have limitations in generalization to unseen environments, prediction fidelity of critical details, and action controllability for flexible application. In this paper, we present Vista, a generalizable driving world model with high fidelity and versatile controllability. Based on a systematic diagnosis of existing methods, we introduce several key ingredients to address these limitations. To accurately predict real-world dynamics at high resolution, we propose two novel losses to promote the learning of moving instances and structural information. We also devise an effective latent replacement approach to inject historical frames as priors for coherent long-horizon rollouts. For action controllability, we incorporate a versatile set of controls from high-level intentions (command, goal point) to low-level maneuvers (trajectory, angle, and speed) through an efficient learning strategy. After large-scale training, the capabilities of Vista can seamlessly generalize to different scenarios. Extensive experiments on multiple datasets show that Vista outperforms the most advanced general-purpose video generator in over 70% of comparisons and surpasses the best-performing driving world model by 55% in FID and 27% in FVD. Moreover, for the first time, we utilize the capacity of Vista itself to establish a generalizable reward for real-world action evaluation without accessing the ground truth actions.
  
![image](https://github.com/user-attachments/assets/6068d5b0-adea-4aee-8eb6-7a06b992b262)

</details>

[📃 arXiv:2311](https://arxiv.org/pdf/2405.17398) | [⌨️ Code](https://github.com/OpenDriveLab/Vista/tree/main?tab=readme-ov-file) | [🌐 Project Page](https://vista-demo.github.io/)

<br>
<br>


## VLA


#### <summary>EMMA: End-to-End Multimodal Model for Autonomous Driving


Authors: Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, Yin Zhou, James Guo, Dragomir Anguelov, Mingxing Tan
<details span>
<summary><b>Abstract</b></summary>
We introduce EMMA, an End-to-end Multimodal Model for Autonomous driving. Built on a multi-modal large language model foundation, EMMA directly maps raw camera sensor data into various driving-specific outputs, including planner trajectories, perception objects, and road graph elements. EMMA maximizes the utility of world knowledge from the pre-trained large language models, by representing all non-sensor inputs (e.g. navigation instructions and ego vehicle status) and outputs (e.g. trajectories and 3D locations) as natural language text. This approach allows EMMA to jointly process various driving tasks in a unified language space, and generate the outputs for each task using task-specific prompts. Empirically, we demonstrate EMMA's effectiveness by achieving state-of-the-art performance in motion planning on nuScenes as well as competitive results on the Waymo Open Motion Dataset (WOMD). EMMA also yields competitive results for camera-primary 3D object detection on the Waymo Open Dataset (WOD). We show that co-training EMMA with planner trajectories, object detection, and road graph tasks yields improvements across all three domains, highlighting EMMA's potential as a generalist model for autonomous driving applications. However, EMMA also exhibits certain limitations: it can process only a small amount of image frames, does not incorporate accurate 3D sensing modalities like LiDAR or radar and is computationally expensive. We hope that our results will inspire further research to mitigate these issues and to further evolve the state of the art in autonomous driving model architectures.

![image](https://github.com/user-attachments/assets/ec7cabb5-3b66-4673-aeaa-8ffe50eddf32)

</details>

[📃 arXiv:2410](https://arxiv.org/pdf/2410.23262) | [⌨️ Code] | [🌐 Project Page](https://waymo.com/blog/2024/10/introducing-emma/)

<br>
<br>


## Others

#### <summary>DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
>*DiffusionDrive+Text Input=VLA?*

Authors: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, Xinggang Wang
<details span>
<summary><b>Abstract</b></summary>
Recently, the diffusion model has emerged as a powerful generative technique for robotic policy learning, capable of modeling multi-mode action distributions. Leveraging its capability for end-to-end autonomous driving is a promising direction. However, the numerous denoising steps in the robotic diffusion policy and the more dynamic, open-world nature of traffic scenes pose substantial challenges for generating diverse driving actions at a real-time speed. To address these challenges, we propose a novel truncated diffusion policy that incorporates prior multi-mode anchors and truncates the diffusion schedule, enabling the model to learn denoising from anchored Gaussian distribution to the multi-mode driving action distribution. Additionally, we design an efficient cascade diffusion decoder for enhanced interaction with conditional scene context. The proposed model, DiffusionDrive, demonstrates 10× reduction in denoising steps compared to vanilla diffusion policy, delivering superior diversity and quality in just 2 steps. On the planning-oriented NAVSIM dataset, with the aligned ResNet-34 backbone, DiffusionDrive achieves 88.1 PDMS without bells and whistles, setting a new record, while running at a real-time speed of 45 FPS on an NVIDIA 4090. Qualitative results on challenging scenarios further confirm that DiffusionDrive can robustly generate diverse plausible driving actions.

![image](https://github.com/user-attachments/assets/6a942c90-72ca-4f2d-8162-0002914dfe20)

</details>

[📃 arXiv:2411](https://arxiv.org/pdf/2411.15139) | [⌨️ Code](https://github.com/hustvl/DiffusionDrive?tab=readme-ov-file#getting-started) | [🌐 Project Page]


#### <summary>TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models
>*for driving scenes?*

Authors: Mark YU, Wenbo Hu, Jinbo Xing, Ying Shan
<details span>
<summary><b>Abstract</b></summary>
We present TrajectoryCrafter, a novel approach to redirect camera trajectories for monocular videos. By disentangling deterministic view transformations from stochastic content generation, our method achieves precise control over user-specified camera trajectories. We propose a novel dual-stream conditional video diffusion model that concurrently integrates point cloud renders and source videos as conditions, ensuring accurate view transformations and coherent 4D content generation. Instead of leveraging scarce multi-view videos, we curate a hybrid training dataset combining web-scale monocular videos with static multi-view datasets, by our innovative double-reprojection strategy, significantly fostering robust generalization across diverse scenes. Extensive evaluations on multi-view and large-scale monocular videos demonstrate the superior performance of our method.

![image](https://github.com/user-attachments/assets/7238ea9a-2793-441d-ac81-7f14cd780ec3)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.05638) | [⌨️ Code](https://github.com/TrajectoryCrafter/TrajectoryCrafter) | [🌐 Project Page](https://trajectorycrafter.github.io/)


#### <summary>Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models
>*Ourkeyinsightisthat the distribution of images degraded by neural rendering artifacts ˜ I resembles the distribution of images xτ originally used to train the diffusion model at a specific noise level τ (Sec. 3). enhance and remove artifacts in rendered novel views caused by underconstrained regions of the 3D representation. Results on Driving Scenes*

Authors: Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling
<details span>
<summary><b>Abstract</b></summary>
Neural Radiance Fields and 3D Gaussian Splatting have revolutionized 3D reconstruction and novel-view synthesis task. However, achieving photorealistic rendering from extreme novel viewpoints remains challenging, as artifacts persist across representations. In this work, we introduce Difix3D+, a novel pipeline designed to enhance 3D reconstruction and novel-view synthesis through single-step diffusion models. At the core of our approach is Difix, a single-step image diffusion model trained to enhance and remove artifacts in rendered novel views caused by underconstrained regions of the 3D representation. Difix serves two critical roles in our pipeline. First, it is used during the reconstruction phase to clean up pseudo-training views that are rendered from the reconstruction and then distilled back into 3D. This greatly enhances underconstrained regions and improves the overall 3D representation quality. More importantly, Difix also acts as a neural enhancer during inference, effectively removing residual artifacts arising from imperfect 3D supervision and the limited capacity of current reconstruction models. Difix3D+ is a general solution, a single model compatible with both NeRF and 3DGS representations, and it achieves an average 2× improvement in FID score over baselines while maintaining 3D consistency.

![image](https://github.com/user-attachments/assets/7f4fa732-0d10-4525-9ae4-79d9f87ff64e)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2411.15540) | [⌨️ Code](https://github.com/HyelinNAM/MotionPrompt) | [🌐 Project Page](https://motionprompt.github.io/)


#### <summary>Optical-Flow Guided Prompt Optimization for Coherent Video Generation
>*enforce temporal consistency in generated videos, enabling smoother, more realistic motion by aligning flows with real-world patterns*

Authors: Hyelin Nam, Jaemin Kim, Dohun Lee, Jong Chul Ye
<details span>
<summary><b>Abstract</b></summary>
While text-to-video diffusion models have made significant strides, many still face challenges in generating videos with temporal consistency. Within diffusion frameworks, guidance techniques have proven effective in enhancing output quality during inference; however, applying these methods to video diffusion models introduces additional complexity of handling computations across entire sequences. To address this, we propose a novel framework called MotionPrompt that guides the video generation process via optical flow. Specifically, we train a discriminator to distinguish optical flow between random pairs of frames from real videos and generated ones. Given that prompts can influence the entire video, we optimize learnable token embeddings during reverse sampling steps by using gradients from a trained discriminator applied to random frame pairs. This approach allows our method to generate visually coherent video sequences that closely reflect natural motion dynamics, without compromising the fidelity of the generated content. We demonstrate the effectiveness of our approach across various models.

![image](https://github.com/user-attachments/assets/5e6b3e71-2922-4a63-ac3c-89a905e00c0d)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.01774) | [⌨️ Code] | [🌐 Project Page](https://research.nvidia.com/labs/toronto-ai/difix3d/)







