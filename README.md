# WorldModel_VLA

<br>

[WorldModel](#WorldModel) | [VLA](#VLA) | [DrivingGaussian](#DrivingGaussian) | [Others](#Others) 

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





#### <summary>From 2D to 3D Cognition: A Brief Survey of General World Models
Authors: Ningwei Xie, Zizi Tian, Lei Yang, Xiao-Ping Zhang, Meng Guo, Jie Li
<details span>
<summary><b>Abstract</b></summary>
World models have garnered increasing attention in the development of artificial general intelligence (AGI), serving as computational frameworks for learning representations of the external world and forecasting future states. While early efforts focused on 2D visual perception and simulation, recent 3D-aware generative world models have demonstrated the ability to synthesize geometrically consistent, interactive 3D environments, marking a shift toward 3D spatial cognition. Despite rapid progress, the field lacks systematic analysis to categorize emerging techniques and clarify their roles in advancing 3D cognitive world models. This survey addresses this need by introducing a conceptual framework, providing a structured and forward-looking review of world models transitioning from 2D perception to 3D cognition. Within this framework, we highlight two key technological drivers, particularly advances in 3D representations and the incorporation of world knowledge, as fundamental pillars. Building on these, we dissect three core cognitive capabilities that underpin 3D world modeling: 3D physical scene generation, 3D spatial reasoning, and 3D spatial interaction. We further examine the deployment of these capabilities in real-world applications, including embodied AI, autonomous driving, digital twin, and gaming/VR. Finally, we identify challenges across data, modeling, and deployment, and outline future directions for advancing more robust and generalizable 3D world models.
  
![image](https://github.com/user-attachments/assets/0c2f51f8-d083-4190-b379-11493c5e124f)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.20134) | [⌨️ Code] | [🌐 Project Page]



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


## DrivingGaussian

#### <summary>Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis
>*selected clear sunny days with uncongested road conditions to drive through each parallel lane in the same direction. Each scene includes three sequences from different lanes, sharing the same start and end positions orthogonal to the road direction, covering approximately 150 meters.*

Authors: Ziqian Ni, Sicong Du, Zhenghua Hou, Chenming Wu, Sheng Yang
<details span>
<summary><b>Abstract</b></summary>
To evaluate end-to-end autonomous driving systems, a simulation environment based on Novel View Synthesis (NVS) techniques is essential, which synthesizes photo-realistic images and point clouds from previously recorded sequences under new vehicle poses, particularly in cross-lane scenarios. Therefore, the development of a multi-lane dataset and benchmark is necessary. While recent synthetic scene-based NVS datasets have been prepared for cross-lane benchmarking, they still lack the realism of captured images and point clouds. To further assess the performance of existing methods based on NeRF and 3DGS, we present the first multi-lane dataset registering parallel scans specifically for novel driving view synthesis dataset derived from real-world scans, comprising 25 groups of associated sequences, including 16,000 front-view images, 64,000 surround-view images, and 16,000 LiDAR frames. All frames are labeled to differentiate moving objects from static elements. Using this dataset, we evaluate the performance of existing approaches in various testing scenarios at different lanes and distances. Additionally, our method provides the solution for solving and assessing the quality of multi-sensor poses for multi-modal data alignment for curating such a dataset in real-world. We plan to continually add new sequences to test the generalization of existing methods across different scenarios.
  
![image](https://github.com/user-attachments/assets/9645b5f2-1eaf-46a2-8a27-d22422201e92)

</details>

[📃 arXiv:2502](https://arxiv.org/pdf/2502.15635) | [⌨️ Code] | [🌐 Project Page](https://nizqleo.github.io/paralane-dataset/)



#### <summary>BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting
>*learnable Bezier curves eliminate the dependence on the accuracy of manual annotations, while effectively representing the complete trajectories*

Authors: Zipei Ma, Junzhe Jiang, Yurui Chen, Li Zhang
<details span>
<summary><b>Abstract</b></summary>
The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose Bézier curve Gaussian splatting (BézierGS), which represents the motion trajectories of dynamic objects using learnable Bézier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that BézierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.

![image](https://github.com/user-attachments/assets/7262238b-9260-4ecb-b034-5457dae095c1)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.22099) | [⌨️ Code](https://github.com/fudan-zvg/BezierGS) | [🌐 Project Page]





## Others

#### <summary>ByTheWay: Boost Your Text-to-Video Generation Model to Higher Quality in a Training-free Way
>*video generation processes exhibiting structurally implausible and temporally inconsistent artifacts demonstrate greater disparity between the temporal attention maps of different decoder blocks.*

Authors: Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang
<details span>
<summary><b>Abstract</b></summary>
The text-to-video (T2V) generation models, offering convenient visual creation, have recently garnered increasing attention. Despite their substantial potential, the generated videos may present artifacts, including structural implausibility, temporal inconsistency, and a lack of motion, often resulting in near-static video. In this work, we have identified a correlation between the disparity of temporal attention maps across different blocks and the occurrence of temporal inconsistencies. Additionally, we have observed that the energy contained within the temporal attention maps is directly related to the magnitude of motion amplitude in the generated videos. Based on these observations, we present ByTheWay, a training-free method to improve the quality of text-to-video generation without introducing additional parameters, augmenting memory or sampling time. Specifically, ByTheWay is composed of two principal components: 1) Temporal Self-Guidance improves the structural plausibility and temporal consistency of generated videos by reducing the disparity between the temporal attention maps across various decoder blocks. 2) Fourier-based Motion Enhancement enhances the magnitude and richness of motion by amplifying the energy of the map. Extensive experiments demonstrate that ByTheWay significantly improves the quality of text-to-video generation with negligible additional cost.

![image](https://github.com/user-attachments/assets/8f0e651e-aec8-4a48-b50d-0835c2add56d)

</details>

[📃 arXiv:2410](https://arxiv.org/pdf/2410.06241) | [⌨️ Code](https://github.com/Bujiazi/ByTheWay) | [🌐 Project Page]



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

![image](https://github.com/user-attachments/assets/5e6b3e71-2922-4a63-ac3c-89a905e00c0d)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2411.15540) | [⌨️ Code](https://github.com/HyelinNAM/MotionPrompt) | [🌐 Project Page](https://motionprompt.github.io/)


#### <summary>Optical-Flow Guided Prompt Optimization for Coherent Video Generation
>*enforce temporal consistency in generated videos, enabling smoother, more realistic motion by aligning flows with real-world patterns*

Authors: Hyelin Nam, Jaemin Kim, Dohun Lee, Jong Chul Ye
<details span>
<summary><b>Abstract</b></summary>
While text-to-video diffusion models have made significant strides, many still face challenges in generating videos with temporal consistency. Within diffusion frameworks, guidance techniques have proven effective in enhancing output quality during inference; however, applying these methods to video diffusion models introduces additional complexity of handling computations across entire sequences. To address this, we propose a novel framework called MotionPrompt that guides the video generation process via optical flow. Specifically, we train a discriminator to distinguish optical flow between random pairs of frames from real videos and generated ones. Given that prompts can influence the entire video, we optimize learnable token embeddings during reverse sampling steps by using gradients from a trained discriminator applied to random frame pairs. This approach allows our method to generate visually coherent video sequences that closely reflect natural motion dynamics, without compromising the fidelity of the generated content. We demonstrate the effectiveness of our approach across various models.

![image](https://github.com/user-attachments/assets/7f4fa732-0d10-4525-9ae4-79d9f87ff64e)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.01774) | [⌨️ Code] | [🌐 Project Page](https://research.nvidia.com/labs/toronto-ai/difix3d/)



#### <summary>GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control
>*Simulating real-world driving scenes along a novel trajectory that is different from captured videos is a cornerstone for training autonomous vehicles. GEN3C can be applied*

Authors: Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao
<details span>
<summary><b>Abstract</b></summary>
We present GEN3C, a generative video model with precise Camera Control and temporal 3D Consistency. Prior video models already generate realistic videos, but they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence. Camera control, if implemented at all, is imprecise, because camera parameters are mere inputs to the neural network which must then infer how the video depends on the camera. In contrast, GEN3C is guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth of seed images or previously generated frames. When generating the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user. Crucially, this means that GEN3C neither has to remember what it previously generated nor does it have to infer the image structure from the camera pose. The model, instead, can focus all its generative power on previously unobserved regions, as well as advancing the scene state to the next frame. Our results demonstrate more precise camera control than prior work, as well as state-of-the-art results in sparse-view novel view synthesis, even in challenging settings such as driving scenes and monocular dynamic video. Results are best viewed in videos.

![image](https://github.com/user-attachments/assets/290c22a1-0f4e-49ff-81e9-1cdc29351bb2)


</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.03751) | [⌨️ Code](https://github.com/nv-tlabs/GEN3C) | [🌐 Project Page](https://research.nvidia.com/labs/toronto-ai/GEN3C/)


#### <summary>Holistic Large-Scale Scene Reconstruction via Mixed Gaussian Splatting
>*......*

Authors: Chuandong Liu, Huijiao Wang, Lei Yu, Gui-Song Xia
<details span>
<summary><b>Abstract</b></summary>
Recent advances in 3D Gaussian Splatting have shown remarkable potential for novel view synthesis. However, most existing large-scale scene reconstruction methods rely on the divide-and-conquer paradigm, which often leads to the loss of global scene information and requires complex parameter tuning due to scene partitioning and local optimization. To address these limitations, we propose MixGS, a novel holistic optimization framework for large-scale 3D scene reconstruction. MixGS models the entire scene holistically by integrating camera pose and Gaussian attributes into a view-aware representation, which is decoded into fine-detailed Gaussians. Furthermore, a novel mixing operation combines decoded and original Gaussians to jointly preserve global coherence and local fidelity. Extensive experiments on large-scale scenes demonstrate that MixGS achieves state-of-the-art rendering quality and competitive speed, while significantly reducing computational requirements, enabling large-scale scene reconstruction training on a single 24GB VRAM GPU. 

![image](https://github.com/user-attachments/assets/e5d83740-2bfa-4b4c-8c4d-ad4ff2b01bd5)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2505.23280) | [⌨️ Code](https://github.com/azhuantou/MixGS) | [🌐 Project Page]


#### <summary>FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
>*VGG feature for feed-forward*

Authors: Chuandong Liu, Huijiao Wang, Lei Yu, Gui-Song Xia
<details span>
<summary><b>Abstract</b></summary>
We present FLARE, a feed-forward model designed to infer high-quality camera poses and 3D geometry from uncalibrated sparse-view images (i.e., as few as 2-8 inputs), which is a challenging yet practical setting in real-world applications. Our solution features a cascaded learning paradigm with camera pose serving as the critical bridge, recognizing its essential role in mapping 3D structures onto 2D image planes. Concretely, FLARE starts with camera pose estimation, whose results condition the subsequent learning of geometric structure and appearance, optimized through the objectives of geometry reconstruction and novel-view synthesis. Utilizing large-scale public datasets for training, our method delivers state-of-the-art performance in the tasks of pose estimation, geometry reconstruction, and novel view synthesis, while maintaining the inference efficiency (i.e., less than 0.5 seconds).

![image](https://github.com/user-attachments/assets/04980959-5daa-4019-b305-e18d69793a40)

</details>

[📃 arXiv:2502](https://arxiv.org/abs/2502.12138) | [⌨️ Code](https://github.com/ant-research/FLARE) | [🌐 Project Page](https://zhanghe3z.github.io/FLARE/)




