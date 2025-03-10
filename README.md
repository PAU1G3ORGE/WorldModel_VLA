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

#### <summary>TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models
>*for driving scenes?*

Authors: Mark YU, Wenbo Hu, Jinbo Xing, Ying Shan
<details span>
<summary><b>Abstract</b></summary>
We present TrajectoryCrafter, a novel approach to redirect camera trajectories for monocular videos. By disentangling deterministic view transformations from stochastic content generation, our method achieves precise control over user-specified camera trajectories. We propose a novel dual-stream conditional video diffusion model that concurrently integrates point cloud renders and source videos as conditions, ensuring accurate view transformations and coherent 4D content generation. Instead of leveraging scarce multi-view videos, we curate a hybrid training dataset combining web-scale monocular videos with static multi-view datasets, by our innovative double-reprojection strategy, significantly fostering robust generalization across diverse scenes. Extensive evaluations on multi-view and large-scale monocular videos demonstrate the superior performance of our method.

![image](https://github.com/user-attachments/assets/7238ea9a-2793-441d-ac81-7f14cd780ec3)

</details>

[📃 arXiv:2503](https://arxiv.org/pdf/2503.05638) | [⌨️ Code](https://github.com/TrajectoryCrafter/TrajectoryCrafter) | [🌐 Project Page](https://trajectorycrafter.github.io/)
