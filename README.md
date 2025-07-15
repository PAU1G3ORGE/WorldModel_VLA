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


#### <summary>Can Test-Time Scaling Improve World Foundation Model?
>*cosmos-predict1*

Authors: Wenyan Cong, Hanqing Zhu, Peihao Wang, Bangya Liu, Dejia Xu, Kevin Wang, David Z. Pan, Yan Wang, Zhiwen Fan, Zhangyang Wang
<details span>
<summary><b>Abstract</b></summary>
World foundation models, which simulate the physical world by predicting future states from current observations and inputs, have become central to many applications in physical intelligence, including autonomous driving and robotics. However, these models require substantial computational resources for pretraining and are further constrained by available data during post-training. As such, scaling computation at test time emerges as both a critical and practical alternative to traditional model enlargement or re-training. In this work, we introduce SWIFT, a test-time scaling framework tailored for WFMs. SWIFT integrates our extensible WFM evaluation toolkit with process-level inference strategies, including fast tokenization, probability-based Top-K pruning, and efficient beam search. Empirical results on the COSMOS model demonstrate that test-time scaling exists even in a compute-optimal way. Our findings reveal that test-time scaling laws hold for WFMs and that SWIFT provides a scalable and effective pathway for improving WFM inference without retraining or increasing model size. 
  
![image](https://github.com/user-attachments/assets/803db78f-e3d8-45fc-887f-b9f1cef262d0)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.24320) | [⌨️ Code](https://github.com/Mia-Cong/SWIFT.git) | [🌐 Project Page]


#### <summary>End-to-End Driving with Online Trajectory Evaluation via BEV World Model
>*3DGS methods need off-board reconstruction and cannot be adopted in our onboard driving setting and lack supervision for future states, while only one future state is available in
 the real-world datasets*

Authors: Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, Zhaoxiang Zhang
<details span>
<summary><b>Abstract</b></summary>
End-to-end autonomous driving has achieved remarkable progress by integrating perception, prediction, and planning into a fully differentiable framework. Yet, to fully realize its potential, an effective online trajectory evaluation is indispensable to ensure safety. By forecasting the future outcomes of a given trajectory, trajectory evaluation becomes much more effective. This goal can be achieved by employing a world model to capture environmental dynamics and predict future states. Therefore, we propose an end-to-end driving framework WoTE, which leverages a BEV World model to predict future BEV states for Trajectory Evaluation. The proposed BEV world model is latency-efficient compared to image-level world models and can be seamlessly supervised using off-the-shelf BEV-space traffic simulators. We validate our framework on both the NAVSIM benchmark and the closed-loop Bench2Drive benchmark based on the CARLA simulator, achieving state-of-the-art performance.
  
![image](https://github.com/user-attachments/assets/8d0df54a-188e-4d77-bf6f-e1aec44854fe)

</details>

[📃 arXiv:2504](https://arxiv.org/pdf/2504.01941) | [⌨️ Code](https://github.com/liyingyanUCAS/WoTE) | [🌐 Project Page]




#### <summary>From 2D to 3D Cognition: A Brief Survey of General World Models
Authors: Ningwei Xie, Zizi Tian, Lei Yang, Xiao-Ping Zhang, Meng Guo, Jie Li
<details span>
<summary><b>Abstract</b></summary>
World models have garnered increasing attention in the development of artificial general intelligence (AGI), serving as computational frameworks for learning representations of the external world and forecasting future states. While early efforts focused on 2D visual perception and simulation, recent 3D-aware generative world models have demonstrated the ability to synthesize geometrically consistent, interactive 3D environments, marking a shift toward 3D spatial cognition. Despite rapid progress, the field lacks systematic analysis to categorize emerging techniques and clarify their roles in advancing 3D cognitive world models. This survey addresses this need by introducing a conceptual framework, providing a structured and forward-looking review of world models transitioning from 2D perception to 3D cognition. Within this framework, we highlight two key technological drivers, particularly advances in 3D representations and the incorporation of world knowledge, as fundamental pillars. Building on these, we dissect three core cognitive capabilities that underpin 3D world modeling: 3D physical scene generation, 3D spatial reasoning, and 3D spatial interaction. We further examine the deployment of these capabilities in real-world applications, including embodied AI, autonomous driving, digital twin, and gaming/VR. Finally, we identify challenges across data, modeling, and deployment, and outline future directions for advancing more robust and generalizable 3D world models.
  
![image](https://github.com/user-attachments/assets/0c2f51f8-d083-4190-b379-11493c5e124f)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.20134) | [⌨️ Code] | [🌐 Project Page]


#### <summary>LongDWM: Cross-Granularity Distillation for Building a Long-Term Driving World Model
>*following work of Vista, decouple the long-term learning into large motion learning (e.g., scene transitions) and small continuous motion learning (e.g., car motions) for error accumulation*
Authors: Xiaodong Wang, Zhirong Wu, Peixi Peng
<details span>
<summary><b>Abstract</b></summary>
Driving world models are used to simulate futures by video generation based on the condition of the current state and actions. However, current models often suffer serious error accumulations when predicting the long-term future, which limits the practical application. Recent studies utilize the Diffusion Transformer (DiT) as the backbone of driving world models to improve learning flexibility. However, these models are always trained on short video clips (high fps and short duration), and multiple roll-out generations struggle to produce consistent and reasonable long videos due to the training-inference gap. To this end, we propose several solutions to build a simple yet effective long-term driving world model. First, we hierarchically decouple world model learning into large motion learning and bidirectional continuous motion learning. Then, considering the continuity of driving scenes, we propose a simple distillation method where fine-grained video flows are self-supervised signals for coarse-grained flows. The distillation is designed to improve the coherence of infinite video generation. The coarse-grained and fine-grained modules are coordinated to generate long-term and temporally coherent videos. In the public benchmark NuScenes, compared with the state-of-the-art front-view model, our model improves FVD by  and reduces inference time by  for the video task of generating 110+ frames.
  
![image](https://github.com/user-attachments/assets/a287dbce-07c2-4c85-8698-3d81bbe6bb4d)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.01546) | [⌨️ Code] | [🌐 Project Page](https://wang-xiaodong1899.github.io/longdwm/)




#### <summary>BEV-VAE: Multi-view Image Generation with Spatial Consistency for Autonomous Driving
>*avoids the ambiguity and lack of depth information introduced by 2D projections of 3D bounding boxes*

Authors: Zeming Chen, Hang Zhao
<details span>
<summary><b>Abstract</b></summary>
Multi-view image generation in autonomous driving demands consistent 3D scene understanding across camera views. Most existing methods treat this problem as a 2D image set generation task, lacking explicit 3D modeling. However, we argue that a structured representation is crucial for scene generation, especially for autonomous driving applications. This paper proposes BEV-VAE for consistent and controllable view synthesis. BEV-VAE first trains a multi-view image variational autoencoder for a compact and unified BEV latent space and then generates the scene with a latent diffusion transformer. BEV-VAE supports arbitrary view generation given camera configurations, and optionally 3D layouts. Experiments on nuScenes and Argoverse 2 (AV2) show strong performance in both 3D consistent reconstruction and generation.
  
![image](https://github.com/user-attachments/assets/46131f2a-cfec-4636-96db-53b049e7bb19)

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.00707) | [⌨️ Code](https://github.com/Czm369/bev-vae) | [🌐 Project Page]






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




#### <summary>DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
>*NOT VLA, DiffusionDrive+Text Input=VLA?*

Authors: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, Xinggang Wang
<details span>
<summary><b>Abstract</b></summary>
Recently, the diffusion model has emerged as a powerful generative technique for robotic policy learning, capable of modeling multi-mode action distributions. Leveraging its capability for end-to-end autonomous driving is a promising direction. However, the numerous denoising steps in the robotic diffusion policy and the more dynamic, open-world nature of traffic scenes pose substantial challenges for generating diverse driving actions at a real-time speed. To address these challenges, we propose a novel truncated diffusion policy that incorporates prior multi-mode anchors and truncates the diffusion schedule, enabling the model to learn denoising from anchored Gaussian distribution to the multi-mode driving action distribution. Additionally, we design an efficient cascade diffusion decoder for enhanced interaction with conditional scene context. The proposed model, DiffusionDrive, demonstrates 10× reduction in denoising steps compared to vanilla diffusion policy, delivering superior diversity and quality in just 2 steps. On the planning-oriented NAVSIM dataset, with the aligned ResNet-34 backbone, DiffusionDrive achieves 88.1 PDMS without bells and whistles, setting a new record, while running at a real-time speed of 45 FPS on an NVIDIA 4090. Qualitative results on challenging scenarios further confirm that DiffusionDrive can robustly generate diverse plausible driving actions.

![image](https://github.com/user-attachments/assets/6a942c90-72ca-4f2d-8162-0002914dfe20)

</details>

[📃 arXiv:2411](https://arxiv.org/pdf/2411.15139) | [⌨️ Code](https://github.com/hustvl/DiffusionDrive?tab=readme-ov-file#getting-started) | [🌐 Project Page]



#### <summary>Enhancing End-to-End Autonomous Driving with Latent World Model
>*NOT VLA, NOT WORLD MODEL, self-supervised method jointly optimizes the current scene feature learning and ego trajectory prediction*

Authors: Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, Tieniu Tan
<details span>
<summary><b>Abstract</b></summary>
In autonomous driving, end-to-end planners directly utilize raw sensor data, enabling them to extract richer scene features and reduce information loss compared to traditional planners. This raises a crucial research question: how can we develop better scene feature representations to fully leverage sensor data in end-to-end driving? Self-supervised learning methods show great success in learning rich feature representations in NLP and computer vision. Inspired by this, we propose a novel self-supervised learning approach using the LAtent World model (LAW) for end-to-end driving. LAW predicts future scene features based on current features and ego trajectories. This self-supervised task can be seamlessly integrated into perception-free and perception-based frameworks, improving scene feature learning and optimizing trajectory prediction. LAW achieves state-of-the-art performance across multiple benchmarks, including real-world open-loop benchmark nuScenes, NAVSIM, and simulator-based closed-loop benchmark CARLA.
  
![image](https://github.com/user-attachments/assets/425631f1-a8e5-4860-b144-57d4e63cc593)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2406.08481) | [⌨️ Code](https://github.com/BraveGroup/LAW) | [🌐 Project Page]

#### <summary>World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model
>*NOT VLA, NOT WORLD MODEL, integrates multi-modal driving intentions with a latent world model (attention layer) to enable rational planning*

Authors: Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, Dongbin Zhao
<details span>
<summary><b>Abstract</b></summary>
End-to-end autonomous driving directly generates planning trajectories from raw sensor data, yet it typically relies on costly perception supervision to extract scene information. A critical research challenge arises: constructing an informative driving world model to enable perception annotation-free, end-to-end planning via self-supervised learning. In this paper, we present World4Drive, an end-to-end autonomous driving framework that employs vision foundation models to build latent world models for generating and evaluating multi-modal planning trajectories. Specifically, World4Drive first extracts scene features, including driving intention and world latent representations enriched with spatial-semantic priors provided by vision foundation models. It then generates multi-modal planning trajectories based on current scene features and driving intentions and predicts multiple intention-driven future states within the latent space. Finally, it introduces a world model selector module to evaluate and select the best trajectory. We achieve perception annotation-free, end-to-end planning through self-supervised alignment between actual future observations and predicted observations reconstructed from the latent space. World4Drive achieves state-of-the-art performance without manual perception annotations on both the open-loop nuScenes and closed-loop NavSim benchmarks, demonstrating an 18.1\% relative reduction in L2 error, 46.7% lower collision rate, and 3.75 faster training convergence.

![image](https://github.com/user-attachments/assets/f93bd193-62e9-4a0f-ad3b-9b4e170191b6)

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.00603) | [⌨️ Code] | [🌐 Project Page]

<br>
<br>


## DrivingGaussian

#### <summary>GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving
> *leveraging 3D Gaussian anchors as volumetric LiDAR points, combined with Ray-based guidance and MAE method*

Authors: Shaoqing Xu, Fang Li, Shengyin Jiang, Ziying Song, Li Liu, Zhi-xin Yang
<details span>
<summary><b>Abstract</b></summary>
Self-supervised learning has made substantial strides in image processing, while visual pre-training for autonomous driving is still in its infancy. Existing methods often focus on learning geometric scene information while neglecting texture or treating both aspects separately, hindering comprehensive scene understanding. In this context, we are excited to introduce GaussianPretrain, a novel pre-training paradigm that achieves a holistic understanding of the scene by uniformly integrating geometric and texture representations. Conceptualizing 3D Gaussian anchors as volumetric LiDAR points, our method learns a deepened understanding of scenes to enhance pre-training performance with detailed spatial structure and texture, achieving that 40.6% faster than NeRF-based method UniPAD with 70% GPU memory only. We demonstrate the effectiveness of GaussianPretrain across multiple 3D perception tasks, showing significant performance improvements, such as a 7.05% increase in NDS for 3D object detection, boosts mAP by 1.9% in HD map construction and 0.8% improvement on Occupancy prediction. These significant gains highlight GaussianPretrain's theoretical innovation and strong practical potential, promoting visual pre-training development for autonomous driving.

![image](https://github.com/user-attachments/assets/00e6bd28-a1e4-4592-a460-42947dfda1c1)

</details>

[📃 arXiv:2411](https://arxiv.org/pdf/2411.12452) | [⌨️ Code](https://github.com/Public-BOTs/GaussianPretrain) | [🌐 Project Page]



#### <summary>Para-Lane: Multi-Lane Dataset Registering Parallel Scans for Benchmarking Novel View Synthesis
>*selected clear sunny days with uncongested road conditions to drive through each parallel lane in the same direction. Each scene includes three sequences from different lanes, sharing the same start and end positions orthogonal to the road direction, covering approximately 150 meters.*

Authors: Ziqian Ni, Sicong Du, Zhenghua Hou, Chenming Wu, Sheng Yang
<details span>
<summary><b>Abstract</b></summary>
To evaluate end-to-end autonomous driving systems, a simulation environment based on Novel View Synthesis (NVS) techniques is essential, which synthesizes photo-realistic images and point clouds from previously recorded sequences under new vehicle poses, particularly in cross-lane scenarios. Therefore, the development of a multi-lane dataset and benchmark is necessary. While recent synthetic scene-based NVS datasets have been prepared for cross-lane benchmarking, they still lack the realism of captured images and point clouds. To further assess the performance of existing methods based on NeRF and 3DGS, we present the first multi-lane dataset registering parallel scans specifically for novel driving view synthesis dataset derived from real-world scans, comprising 25 groups of associated sequences, including 16,000 front-view images, 64,000 surround-view images, and 16,000 LiDAR frames. All frames are labeled to differentiate moving objects from static elements. Using this dataset, we evaluate the performance of existing approaches in various testing scenarios at different lanes and distances. Additionally, our method provides the solution for solving and assessing the quality of multi-sensor poses for multi-modal data alignment for curating such a dataset in real-world. We plan to continually add new sequences to test the generalization of existing methods across different scenarios.
  
![image](https://github.com/user-attachments/assets/9645b5f2-1eaf-46a2-8a27-d22422201e92)

</details>

[📃 arXiv:2502](https://arxiv.org/pdf/2502.15635) | [⌨️ Code] | [🌐 Project Page](https://nizqleo.github.io/paralane-dataset/)

#### <summary>EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis
>*feed-forward driving scenes*

Authors: Sheng Miao, Jiaxin Huang, Dongfeng Bai, Xu Yan, Hongyu Zhou, Yue Wang, Bingbing Liu, Andreas Geiger, Yiyi Liao
<details span>
<summary><b>Abstract</b></summary>
Novel view synthesis of urban scenes is essential for autonomous driving-related this http URL NeRF and 3DGS-based methods show promising results in achieving photorealistic renderings but require slow, per-scene optimization. We introduce EVolSplat, an efficient 3D Gaussian Splatting model for urban scenes that works in a feed-forward manner. Unlike existing feed-forward, pixel-aligned 3DGS methods, which often suffer from issues like multi-view inconsistencies and duplicated content, our approach predicts 3D Gaussians across multiple frames within a unified volume using a 3D convolutional network. This is achieved by initializing 3D Gaussians with noisy depth predictions, and then refining their geometric properties in 3D space and predicting color based on 2D textures. Our model also handles distant views and the sky with a flexible hemisphere background model. This enables us to perform fast, feed-forward reconstruction while achieving real-time rendering. Experimental evaluations on the KITTI-360 and Waymo datasets show that our method achieves state-of-the-art quality compared to existing feed-forward 3DGS- and NeRF-based methods.
  
<img width="1744" height="823" alt="image" src="https://github.com/user-attachments/assets/45cd7d9d-a4ad-4c4b-8e43-3d3ff3236677" />

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.20168) | [⌨️ Code](https://github.com/Miaosheng1/EVolSplat) | [🌐 Project Page](https://xdimlab.github.io/EVolSplat/)




#### <summary>BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting
>*learnable Bezier curves eliminate the dependence on the accuracy of manual annotations, while effectively representing the complete trajectories*

Authors: Zipei Ma, Junzhe Jiang, Yurui Chen, Li Zhang
<details span>
<summary><b>Abstract</b></summary>
The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose Bézier curve Gaussian splatting (BézierGS), which represents the motion trajectories of dynamic objects using learnable Bézier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that BézierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.

![image](https://github.com/user-attachments/assets/7262238b-9260-4ecb-b034-5457dae095c1)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.22099) | [⌨️ Code](https://github.com/fudan-zvg/BezierGS) | [🌐 Project Page]


#### <summary>RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors
>*Reward Map for diffusion output*

Authors: Sicong Du, Jiarun Liu, Qifeng Chen, Hao-Xiang Chen, Tai-Jiang Mu, Sheng Yang
<details span>
<summary><b>Abstract</b></summary>
A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality.

![image](https://github.com/user-attachments/assets/8a8764c2-0c9c-42d6-b6e9-dbdce4580327)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.22800) | [⌨️ Code](https://github.com/CN-ADLab/RGE-GS) | [🌐 Project Page]

#### <summary>Feed-Forward SceneDINO for Unsupervised Semantic Scene Completion
>*3D Feature Field for semantic scene completion*

Authors: Aleksandar Jevtić, Christoph Reich, Felix Wimbauer, Oliver Hahn, Christian Rupprecht, Stefan Roth, Daniel Cremers
<details span>
<summary><b>Abstract</b></summary>
Semantic scene completion (SSC) aims to infer both the 3D geometry and semantics of a scene from single images. In contrast to prior work on SSC that heavily relies on expensive ground-truth annotations, we approach SSC in an unsupervised setting. Our novel method, SceneDINO, adapts techniques from self-supervised representation learning and 2D unsupervised scene understanding to SSC. Our training exclusively utilizes multi-view consistency self-supervision without any form of semantic or geometric ground truth. Given a single input image, SceneDINO infers the 3D geometry and expressive 3D DINO features in a feed-forward manner. Through a novel 3D feature distillation approach, we obtain unsupervised 3D semantics. In both 3D and 2D unsupervised scene understanding, SceneDINO reaches state-of-the-art segmentation accuracy. Linear probing our 3D features matches the segmentation accuracy of a current supervised SSC approach. Additionally, we showcase the domain generalization and multi-view consistency of SceneDINO, taking the first steps towards a strong foundation for single image 3D scene understanding.

<img width="1812" height="286" alt="image" src="https://github.com/user-attachments/assets/22bd8019-c4dc-4c7d-8153-5e0d18adf9b6" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.06230) | [⌨️ Code](https://github.com/tum-vision/scenedino) | [🌐 Project Page](https://visinf.github.io/scenedino/)


<br>
<br>



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


#### <summary>VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory
> *CUT3R for Retrieved, retrieved image for video generation*

Authors: Runjia Li, Philip Torr, Andrea Vedaldi, Tomas Jakab
<details span>
<summary><b>Abstract</b></summary>
We propose a novel memory mechanism to build video generators that can explore environments interactively. Similar results have previously been achieved by out-painting 2D views of the scene while incrementally reconstructing its 3D geometry, which quickly accumulates errors, or by video generators with a short context window, which struggle to maintain scene coherence over the long term. To address these limitations, we introduce Surfel-Indexed View Memory (VMem), a mechanism that remembers past views by indexing them geometrically based on the 3D surface elements (surfels) they have observed. VMem enables the efficient retrieval of the most relevant past views when generating new ones. By focusing only on these relevant views, our method produces consistent explorations of imagined environments at a fraction of the computational cost of using all past views as context. We evaluate our approach on challenging long-term scene synthesis benchmarks and demonstrate superior performance compared to existing methods in maintaining scene coherence and camera control.

![image](https://github.com/user-attachments/assets/ee8fda60-532e-48f6-99ec-caf56394ff11)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.18903) | [⌨️ Code](https://github.com/runjiali-rl/vmem) | [🌐 Project Page](https://v-mem.github.io/)



#### <summary>VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding
> *3D points vote for 2d center, 3D point vote for 2d plane in driving scenes?*

Authors: Minchao Jiang, Shunyu Jia, Jiaming Gu, Xiaoyuan Lu, Guangming Zhu, Anqi Dong, Liang Zhang
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has become horsepower in high-quality, real-time rendering for novel view synthesis of 3D scenes. However, existing methods focus primarily on geometric and appearance modeling, lacking deeper scene understanding while also incurring high training costs that complicate the originally streamlined differentiable rendering pipeline. To this end, we propose VoteSplat, a novel 3D scene understanding framework that integrates Hough voting with 3DGS. Specifically, Segment Anything Model (SAM) is utilized for instance segmentation, extracting objects, and generating 2D vote maps. We then embed spatial offset vectors into Gaussian primitives. These offsets construct 3D spatial votes by associating them with 2D image votes, while depth distortion constraints refine localization along the depth axis. For open-vocabulary object localization, VoteSplat maps 2D image semantics to 3D point clouds via voting points, reducing training costs associated with high-dimensional CLIP features while preserving semantic unambiguity. Extensive experiments demonstrate effectiveness of VoteSplat in open-vocabulary 3D instance localization, 3D point cloud understanding, click-based 3D object localization, hierarchical segmentation, and ablation studies. 

![image](https://github.com/user-attachments/assets/77480463-446c-432f-8a8f-6cb12d000a6a)

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.22799) | [⌨️ Code] | [🌐 Project Page](https://sy-ja.github.io/votesplat/)


#### <summary>FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model
> *features from the right image can largely enhance the smoothness and identity preservation of the image transitions, substitute the original K and V with those derived from the right images*

Authors: Yukang Cao, Chenyang Si, Jinghao Wang, Ziwei Liu
<details span>
<summary><b>Abstract</b></summary>
We present FreeMorph, the first tuning-free method for image morphing that accommodates inputs with different semantics or layouts. Unlike existing methods that rely on finetuning pre-trained diffusion models and are limited by time constraints and semantic/layout discrepancies, FreeMorph delivers high-fidelity image morphing without requiring per-instance training. Despite their efficiency and potential, tuning-free methods face challenges in maintaining high-quality results due to the non-linear nature of the multi-step denoising process and biases inherited from the pre-trained diffusion model. In this paper, we introduce FreeMorph to address these challenges by integrating two key innovations. 1) We first propose a guidance-aware spherical interpolation design that incorporates explicit guidance from the input images by modifying the self-attention modules, thereby addressing identity loss and ensuring directional transitions throughout the generated sequence. 2) We further introduce a step-oriented variation trend that blends self-attention modules derived from each input image to achieve controlled and consistent transitions that respect both inputs. Our extensive evaluations demonstrate that FreeMorph outperforms existing methods, being 10x ~ 50x faster and establishing a new state-of-the-art for image morphing.

![image](https://github.com/user-attachments/assets/61b4e2bc-d1b2-480b-ba96-05b08bb6e96d)

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.01953) | [⌨️ Code](https://github.com/yukangcao/FreeMorph) | [🌐 Project Page](https://yukangcao.github.io/FreeMorph/)





#### <summary>Frequency Regulation for Exposure Bias Mitigation in Diffusion Models
> *The distribution is changed during frequency regulation. Is this correct?*

Authors: Meng Yu, Kun Zhan
<details span>
<summary><b>Abstract</b></summary>
Diffusion models exhibit impressive generative capabilities but are significantly impacted by exposure bias. In this paper, we make a key observation: the energy of the predicted noisy images decreases during the diffusion process. Building on this, we identify two important findings: 1) The reduction in energy follows distinct patterns in the low-frequency and high-frequency subbands; 2) This energy reduction results in amplitude variations between the network-reconstructed clean data and the real clean data. Based on the first finding, we introduce a frequency-domain regulation mechanism utilizing wavelet transforms, which separately adjusts the low- and high-frequency subbands. Leveraging the second insight, we provide a more accurate analysis of exposure bias in the two subbands. Our method is training-free and plug-and-play, significantly improving the generative quality of various diffusion models and providing a robust solution to exposure bias across different model architectures. 

<img width="978" height="505" alt="image" src="https://github.com/user-attachments/assets/cd3d14f3-34df-40dc-9080-1322491d7292" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.10072) | [⌨️ Code] | [🌐 Project Page]



#### <summary>Improving Remote Sensing Classification using Topological Data Analysis and Convolutional Neural Networks
> *Topological data analysis (TDA)*

Authors: Aaryam Sharma
<details span>
<summary><b>Abstract</b></summary>
Topological data analysis (TDA) is a relatively new field that is gaining rapid adoption due to its robustness and ability to effectively describe complex datasets by quantifying geometric information. In imaging contexts, TDA typically models data as filtered cubical complexes from which we can extract discriminative features using persistence homology. Meanwhile, convolutional neural networks (CNNs) have been shown to be biased towards texture based local features. To address this limitation, we propose a TDA feature engineering pipeline and a simple method to integrate topological features with deep learning models on remote sensing classification. Our method improves the performance of a ResNet18 model on the EuroSAT dataset by 1.44% achieving 99.33% accuracy, which surpasses all previously reported single-model accuracies, including those with larger architectures, such as ResNet50 (2x larger) and XL Vision Transformers (197x larger). We additionally show that our method's accuracy is 1.82% higher than our ResNet18 baseline on the RESISC45 dataset. To our knowledge, this is the first application of TDA features in satellite scene classification with deep learning. This demonstrates that TDA features can be integrated with deep learning models, even on datasets without explicit topological structures, thereby increasing the applicability of TDA. A clean implementation of our method will be made publicly available upon publication.

<img width="1955" height="807" alt="image" src="https://github.com/user-attachments/assets/44776131-9605-408a-b8f3-e42d1f3130f7" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.10381) | [⌨️ Code] | [🌐 Project Page]

