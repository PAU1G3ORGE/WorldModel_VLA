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


#### <summary>Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models
>*continuous and discrete prediction losses on a fair common ground and find a clear advantage in favor of continuous modeling. new metric: map both real and generated videos to pose sequences using the inverse dynamics model VGGT, measure distances between real and generated trajectories.*

Authors: Arian Mousakhan, Sudhanshu Mittal, Silvio Galesso, Karim Farid, Thomas Brox
<details span>
<summary><b>Abstract</b></summary>
Existing world models for autonomous driving struggle with long-horizon generation and generalization to challenging scenarios. In this work, we develop a model using simple design choices, and without additional supervision or sensors, such as maps, depth, or multiple cameras. We show that our model yields state-of-the-art performance, despite having only 469M parameters and being trained on 280h of video data. It particularly stands out in difficult scenarios like turning maneuvers and urban traffic. We test whether discrete token models possibly have advantages over continuous models based on flow matching. To this end, we set up a hybrid tokenizer that is compatible with both approaches and allows for a side-by-side comparison. Our study concludes in favor of the continuous autoregressive model, which is less brittle on individual design choices and more powerful than the model built on discrete tokens.
  
<img width="1459" height="732" alt="image" src="https://github.com/user-attachments/assets/d1aee24d-9d40-4ba9-a025-516991811413" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.13162v1) | [⌨️ Code] | [🌐 Project Page](https://lmb-freiburg.github.io/orbis.github.io/)



#### <summary>World Model-Based End-to-End Scene Generation for Accident Anticipation in Autonomous Driving
>*propose a driving scene generation framework for data augmentation in accident anticipation and release the Anticipation of Traffic Accident (AoTA) dataset*

Authors: Yanchen Guan, Haicheng Liao, Chengyue Wang, Xingcheng Liu, Jiaxun Zhang, Zhenning Li
<details span>
<summary><b>Abstract</b></summary>
Reliable anticipation of traffic accidents is essential for advancing autonomous driving systems. However, this objective is limited by two fundamental challenges: the scarcity of diverse, high-quality training data and the frequent absence of crucial object-level cues due to environmental disruptions or sensor deficiencies. To tackle these issues, we propose a comprehensive framework combining generative scene augmentation with adaptive temporal reasoning. Specifically, we develop a video generation pipeline that utilizes a world model guided by domain-informed prompts to create high-resolution, statistically consistent driving scenarios, particularly enriching the coverage of edge cases and complex interactions. In parallel, we construct a dynamic prediction model that encodes spatio-temporal relationships through strengthened graph convolutions and dilated temporal operators, effectively addressing data incompleteness and transient visual noise. Furthermore, we release a new benchmark dataset designed to better capture diverse real-world driving risks. Extensive experiments on public and newly released datasets confirm that our framework enhances both the accuracy and lead time of accident anticipation, offering a robust solution to current data and modeling limitations in safety-critical autonomous driving applications.
  
<img width="976" height="474" alt="image" src="https://github.com/user-attachments/assets/e2e5eab7-dc32-476c-9e38-b586a3dfee3b" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.12762) | [⌨️ Code] | [🌐 Project Page]


#### <summary>HERO: Hierarchical Extrapolation and Refresh for Efficient World Models
>*video diffusion for world model: shallow layers exhibit high temporal variability, while deeper layers yield more stable feature representations*

Authors: Quanjian Song, Xinyu Wang, Donghao Zhou, Jingyu Lin, Cunjian Chen, Yue Ma, Xiu Li
<details span>
<summary><b>Abstract</b></summary>
Generation-driven world models create immersive virtual environments but suffer slow inference due to the iterative nature of diffusion models. While recent advances have improved diffusion model efficiency, directly applying these techniques to world models introduces limitations such as quality degradation. In this paper, we present HERO, a training-free hierarchical acceleration framework tailored for efficient world models. Owing to the multi-modal nature of world models, we identify a feature coupling phenomenon, wherein shallow layers exhibit high temporal variability, while deeper layers yield more stable feature representations. Motivated by this, HERO adopts hierarchical strategies to accelerate inference: (i) In shallow layers, a patch-wise refresh mechanism efficiently selects tokens for recomputation. With patch-wise sampling and frequency-aware tracking, it avoids extra metric computation and remain compatible with FlashAttention. (ii) In deeper layers, a linear extrapolation scheme directly estimates intermediate features. This completely bypasses the computations in attention modules and feed-forward networks. Our experiments show that HERO achieves a 1.73 speedup with minimal quality degradation, significantly outperforming existing diffusion acceleration methods.

<img width="1613" height="329" alt="image" src="https://github.com/user-attachments/assets/d0e4d47f-d780-42ae-9908-3f9353b21101" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.17588) | [⌨️ Code] | [🌐 Project Page]


#### <summary>GWM: Towards Scalable Gaussian World Models for Robotic Manipulation
>*3D Gaussian VAE*

Authors: Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang
<details span>
<summary><b>Abstract</b></summary>
Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.

<img width="1961" height="505" alt="image" src="https://github.com/user-attachments/assets/5aea4b20-d22a-4261-865f-d174fdbf0867" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.17600) | [⌨️ Code] | [🌐 Project Page](https://gaussian-world-model.github.io/)


#### <summary>Realistic and Controllable 3D Gaussian-Guided Object Editing for Driving Video Generation
>*3D Gaussian objects insert into diffusion model*

Authors: Jiusi Li, Jackson Jiang, Jinyu Miao, Miao Long, Tuopu Wen, Peijin Jia, Shengxiang Liu, Chunlei Yu, Maolin Liu, Yuzhan Cai, Kun Jiang, Mengmeng Yang, Diange Yang
<details span>
<summary><b>Abstract</b></summary>
Corner cases are crucial for training and validating autonomous driving systems, yet collecting them from the real world is often costly and hazardous. Editing objects within captured sensor data offers an effective alternative for generating diverse scenarios, commonly achieved through 3D Gaussian Splatting or image generative models. However, these approaches often suffer from limited visual fidelity or imprecise pose control. To address these issues, we propose G^2Editor, a framework designed for photorealistic and precise object editing in driving videos. Our method leverages a 3D Gaussian representation of the edited object as a dense prior, injected into the denoising process to ensure accurate pose control and spatial consistency. A scene-level 3D bounding box layout is employed to reconstruct occluded areas of non-target objects. Furthermore, to guide the appearance details of the edited object, we incorporate hierarchical fine-grained features as additional conditions during generation. Experiments on the Waymo Open Dataset demonstrate that G^2Editor effectively supports object repositioning, insertion, and deletion within a unified framework, outperforming existing methods in both pose controllability and visual quality, while also benefiting downstream data-driven tasks.

<img width="2121" height="919" alt="image" src="https://github.com/user-attachments/assets/c28dd784-5bbe-4b42-8c0d-ec1275cf6ae2" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.20471) | [⌨️ Code] | [🌐 Project Page]


#### <summary>UrbanTwin: High-Fidelity Synthetic Replicas of Roadside Lidar Datasets
>*Sensor Simulation Pipeline*

Authors: Zhengqing Chen, Ruohong Mei, Xiaoyang Guo, Qingjie Wang, Yubin Hu, Wei Yin, Weiqiang Ren, Qian Zhang
<details span>
<summary><b>Abstract</b></summary>
In the field of autonomous driving, sensor simulation is essential for generating rare and diverse scenarios that are difficult to capture in real-world environments. Current solutions fall into two categories: 1) CG-based methods, such as CARLA, which lack diversity and struggle to scale to the vast array of rare cases required for robust perception training; and 2) learning-based approaches, such as NeuSim, which are limited to specific object categories (vehicles) and require extensive multi-sensor data, hindering their applicability to generic objects. To address these limitations, we propose a scalable real2sim2real system that leverages 3D generation to automate asset mining, generation, and rare-case data synthesis.

<img width="1515" height="774" alt="image" src="https://github.com/user-attachments/assets/1974e415-e0b3-470f-9eb8-90193e34752e" />

</details>

[📃 arXiv:2509](https://arxiv.org/pdf/2509.06798) | [⌨️ Code] | [🌐 Project Page]


#### <summary>UrbanTwin: High-Fidelity Synthetic Replicas of Roadside Lidar Datasets
>*Sensor Simulation Pipeline*

Authors: Muhammad Shahbaz, Shaurya Agarwal
<details span>
<summary><b>Abstract</b></summary>
This article presents UrbanTwin datasets - high-fidelity, realistic replicas of three public roadside lidar datasets: LUMPI, V2X-Real-IC, and TUMTraf-I. Each UrbanTwin dataset contains 10K annotated frames corresponding to one of the public datasets. Annotations include 3D bounding boxes, instance segmentation labels, and tracking IDs for six object classes, along with semantic segmentation labels for nine classes. These datasets are synthesized using emulated lidar sensors within realistic digital twins, modeled based on surrounding geometry, road alignment at lane level, and the lane topology and vehicle movement patterns at intersections of the actual locations corresponding to each real dataset. Due to the precise digital twin modeling, the synthetic datasets are well aligned with their real counterparts, offering strong standalone and augmentative value for training deep learning models on tasks such as 3D object detection, tracking, and semantic and instance segmentation. We evaluate the alignment of the synthetic replicas through statistical and structural similarity analysis with real data, and further demonstrate their utility by training 3D object detection models solely on synthetic data and testing them on real, unseen data. The high similarity scores and improved detection performance, compared to the models trained on real data, indicate that the UrbanTwin datasets effectively enhance existing benchmark datasets by increasing sample size and scene diversity. In addition, the digital twins can be adapted to test custom scenarios by modifying the design and dynamics of the simulations. To our knowledge, these are the first digitally synthesized datasets that can replace in-domain real-world datasets for lidar perception tasks. 

<img width="1375" height="705" alt="image" src="https://github.com/user-attachments/assets/170749f0-2a14-4a09-9003-d2dc0cc22538" />

</details>

[📃 arXiv:2509](https://arxiv.org/pdf/2509.06781) | [⌨️ Code] | [🌐 Project Page]




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

#### <summary>Bench2ADVLM: A Closed-Loop Benchmark for Vision-language Models in Autonomous Driving

Authors: Tianyuan Zhang, Ting Jin, Lu Wang, Jiangfan Liu, Siyuan Liang, Mingchuan Zhang, Aishan Liu, Xianglong Liu
<details span>
<summary><b>Abstract</b></summary>
Vision-Language Models (VLMs) have recently emerged as a promising paradigm in autonomous driving (AD). However, current performance evaluation protocols for VLM-based AD systems (ADVLMs) are predominantly confined to open-loop settings with static inputs, neglecting the more realistic and informative closed-loop setting that captures interactive behavior, feedback resilience, and real-world safety. To address this, we introduce Bench2ADVLM, a unified hierarchical closed-loop evaluation framework for real-time, interactive assessment of ADVLMs across both simulation and physical platforms. Inspired by dual-process theories of cognition, we first adapt diverse ADVLMs to simulation environments via a dual-system adaptation architecture. In this design, heterogeneous high-level driving commands generated by target ADVLMs (fast system) are interpreted by a general-purpose VLM (slow system) into standardized mid-level control actions suitable for execution in simulation. To bridge the gap between simulation and reality, we design a physical control abstraction layer that translates these mid-level actions into low-level actuation signals, enabling, for the first time, closed-loop testing of ADVLMs on physical vehicles. To enable more comprehensive evaluation, Bench2ADVLM introduces a self-reflective scenario generation module that automatically explores model behavior and uncovers potential failure modes for safety-critical scenario generation. Overall, Bench2ADVLM establishes a hierarchical evaluation pipeline that seamlessly integrates high-level abstract reasoning, mid-level simulation actions, and low-level real-world execution. Experiments on diverse scenarios across multiple state-of-the-art ADVLMs and physical platforms validate the diagnostic strength of our framework, revealing that existing ADVLMs still exhibit limited performance under closed-loop conditions.

<img width="1880" height="958" alt="image" src="https://github.com/user-attachments/assets/50e47637-3119-4a90-a661-79a604a5dec7" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.02028) | [⌨️ Code] | [🌐 Project Page]

#### <summary>IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model

Authors: Anqing Jiang, Yu Gao, Yiru Wang, Zhigang Sun, Shuo Wang, Yuwen Heng, Hao Sun, Shichen Tang, Lijuan Zhu, Jinhao Chai, Jijun Wang, Zichong Gu, Hao Jiang, Li Sun
<details span>
<summary><b>Abstract</b></summary>
Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving.

<img width="2146" height="698" alt="image" src="https://github.com/user-attachments/assets/1a7db595-439f-4702-9af6-c3c06c25ffc3" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.06571) | [⌨️ Code](https://github.com/IRL-VLA/IRL-VLA) | [🌐 Project Page]


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


#### <summary>AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
>*a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions. modeling object Gaussians' motions by deforming their
 positions µ and rotations R over time using B-splines and trigonometric functions. a regularization loss to ensure that nearby object Gaussians G∈Ω obj exhibit similar deformations in position and temporal visibility mask*

Authors: Jiawei Xu, Kai Deng, Zexin Fan, Shenlong Wang, Jin Xie, Jian Yang
<details span>
<summary><b>Abstract</b></summary>
Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.

<img width="1812" height="286" alt="image" src="https://github.com/user-attachments/assets/22bd8019-c4dc-4c7d-8153-5e0d18adf9b6" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.12137) | [⌨️ Code](https://github.com/JiaweiXu8/AD-GS) | [🌐 Project Page](https://jiaweixu8.github.io/AD-GS-web/)


#### <summary>LidarPainter: One-Step Away From Any Lidar View To Novel Guidance
>*One-Step Diffusion, but struct still is mess*

Authors: Yuzhou Ji, Ke Ma, Hong Cai, Anchun Zhang, Lizhuang Ma, Xin Tan
<details span>
<summary><b>Abstract</b></summary>
Dynamic driving scene reconstruction is of great importance in fields like digital twin system and autonomous driving simulation. However, unacceptable degradation occurs when the view deviates from the input trajectory, leading to corrupted background and vehicle models. To improve reconstruction quality on novel trajectory, existing methods are subject to various limitations including inconsistency, deformation, and time consumption. This paper proposes LidarPainter, a one-step diffusion model that recovers consistent driving views from sparse LiDAR condition and artifact-corrupted renderings in real-time, enabling high-fidelity lane shifts in driving scene reconstruction. Extensive experiments show that LidarPainter outperforms state-of-the-art methods in speed, quality and resource efficiency, specifically 7 x faster than StreetCrafter with only one fifth of GPU memory required. LidarPainter also supports stylized generation using text prompts such as "foggy" and "night", allowing for a diverse expansion of the existing asset library.

<img width="2030" height="786" alt="image" src="https://github.com/user-attachments/assets/c8149b4f-e66d-4158-acd1-7d8d8f003f30" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.12114) | [⌨️ Code] | [🌐 Project Page]


#### <summary>GS-Occ3D: Scaling Vision-only Occupancy Reconstruction for Autonomous Driving with Gaussian Splatting
>*Ground Reconstruction.*

Authors: Baijun Ye, Minghui Qin, Saining Zhang, Moonjun Gong, Shaoting Zhu, Zebang Shen, Luan Zhang, Lu Zhang, Hao Zhao, Hang Zhao
<details span>
<summary><b>Abstract</b></summary>
Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for scalable auto-labeling.

<img width="1555" height="803" alt="image" src="https://github.com/user-attachments/assets/a7b8f8c6-7c19-4f7d-bf1e-bb5a3de446c0" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.19451) | [⌨️ Code] | [🌐 Project Page](https://gs-occ3d.github.io/)


#### <summary>RoGs: Large Scale Road Surface Reconstruction with Meshgrid Gaussian
>*vehicle poses are commonly parallel to the road*

Authors: Zhiheng Feng, Wenhua Wu, Tianchen Deng, Hesheng Wang
<details span>
<summary><b>Abstract</b></summary>
Road surface reconstruction plays a crucial role in autonomous driving, which can be used for road lane perception and autolabeling. Recently, mesh-based road surface reconstruction algorithms have shown promising reconstruction results. However, these mesh-based methods suffer from slow speed and poor reconstruction quality. To address these limitations, we propose a novel large-scale road surface reconstruction approach with meshgrid Gaussian, named RoGs. Specifically, we model the road surface by placing Gaussian surfels in the vertices of a uniformly distributed square mesh, where each surfel stores color, semantic, and geometric information. This square mesh-based layout covers the entire road with fewer Gaussian surfels and reduces the overlap between Gaussian surfels during training. In addition, because the road surface has no thickness, 2D Gaussian surfel is more consistent with the physical reality of the road surface than 3D Gaussian sphere. Then, unlike previous initialization methods that rely on point clouds, we introduce a vehicle pose-based initialization method to initialize the height and rotation of the Gaussian surfel. Thanks to this meshgrid Gaussian modeling and pose-based initialization, our method achieves significant speedups while improving reconstruction quality. We obtain excellent results in reconstruction of road surfaces in a variety of challenging real-world scenes.
  
<img width="1418" height="557" alt="image" src="https://github.com/user-attachments/assets/c0212ffd-2d96-45d9-99d4-ef76c24bccf4" />


</details>

[📃 arXiv:2505](https://arxiv.org/pdf/2405.14342) | [⌨️ Code] | [🌐 Project Page]



#### <summary>MagicRoad: Semantic-Aware 3D Road Surface Reconstruction via Obstacle Inpainting
>*video diffusion*

Authors: Xingyue Peng, Yuandong Lyu, Lang Zhang, Jian Zhu, Songtao Wang, Jiaxin Deng, Songxin Lu, Weiliang Ma, Dangen She, Peng Jia, XianPeng Lang
<details span>
<summary><b>Abstract</b></summary>
Road surface reconstruction is essential for autonomous driving, supporting centimeter-accurate lane perception and high-definition mapping in complex urban environments. While recent methods based on mesh rendering or 3D Gaussian splatting (3DGS) achieve promising results under clean and static conditions, they remain vulnerable to occlusions from dynamic agents, visual clutter from static obstacles, and appearance degradation caused by lighting and weather changes. We present a robust reconstruction framework that integrates occlusion-aware 2D Gaussian surfels with semantic-guided color enhancement to recover clean, consistent road surfaces. Our method leverages a planar-adapted Gaussian representation for efficient large-scale modeling, employs segmentation-guided video inpainting to remove both dynamic and static foreground objects, and enhances color coherence via semantic-aware correction in HSV space. Extensive experiments on urban-scale datasets demonstrate that our framework produces visually coherent and geometrically faithful reconstructions, significantly outperforming prior methods under real-world conditions.
  
<img width="1162" height="487" alt="image" src="https://github.com/user-attachments/assets/cea82fde-7d83-403c-bc5d-ea67715e0631" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.23340) | [⌨️ Code] | [🌐 Project Page]


#### <summary>CRUISE: Cooperative Reconstruction and Editing in V2X Scenarios using Gaussian Splatting

Authors: Haoran Xu, Saining Zhang, Peishuo Li, Baijun Ye, Xiaoxue Chen, Huan-ang Gao, Jv Zheng, Xiaowei Song, Ziqiao Peng, Run Miao, Jinrang Jia, Yifeng Shi, Guangqi Yi, Hang Zhao, Hao Tang, Hongyang Li, Kaicheng Yu, Hao Zhao
<details span>
<summary><b>Abstract</b></summary>
Vehicle-to-everything (V2X) communication plays a crucial role in autonomous driving, enabling cooperation between vehicles and infrastructure. While simulation has significantly contributed to various autonomous driving tasks, its potential for data generation and augmentation in V2X scenarios remains underexplored. In this paper, we introduce CRUISE, a comprehensive reconstruction-and-synthesis framework designed for V2X driving environments. CRUISE employs decomposed Gaussian Splatting to accurately reconstruct real-world scenes while supporting flexible editing. By decomposing dynamic traffic participants into editable Gaussian representations, CRUISE allows for seamless modification and augmentation of driving scenes. Furthermore, the framework renders images from both ego-vehicle and infrastructure views, enabling large-scale V2X dataset augmentation for training and evaluation. Our experimental results demonstrate that: 1) CRUISE reconstructs real-world V2X driving scenes with high fidelity; 2) using CRUISE improves 3D detection across ego-vehicle, infrastructure, and cooperative views, as well as cooperative 3D tracking on the V2X-Seq benchmark; and 3) CRUISE effectively generates challenging corner cases.

<img width="1878" height="468" alt="image" src="https://github.com/user-attachments/assets/ed85c9af-fd5c-4740-9f58-08fe227109f1" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.18473) | [⌨️ Code](https://github.com/SainingZhang/CRUISE) | [🌐 Project Page]




#### <summary>I2V-GS: Infrastructure-to-Vehicle View Transformation with Gaussian Splatting for Autonomous Driving Data Generation
>*For a fair comparison, we train the 3DGS-based methods using the first frame and 4DGS-based methods using the first 10 frames ?*

Authors: Jialei Chen, Wuhao Xu, Sipeng He, Baoru Huang, Dongchun Ren
<details span>
<summary><b>Abstract</b></summary>
Vast and high-quality data are essential for end-to-end autonomous driving systems. However, current driving data is mainly collected by vehicles, which is expensive and inefficient. A potential solution lies in synthesizing data from real-world images. Recent advancements in 3D reconstruction demonstrate photorealistic novel view synthesis, highlighting the potential of generating driving data from images captured on the road. This paper introduces a novel method, I2V-GS, to transfer the Infrastructure view To the Vehicle view with Gaussian Splatting. Reconstruction from sparse infrastructure viewpoints and rendering under large view transformations is a challenging problem. We adopt the adaptive depth warp to generate dense training views. To further expand the range of views, we employ a cascade strategy to inpaint warped images, which also ensures inpainting content is consistent across views. To further ensure the reliability of the diffusion model, we utilize the cross-view information to perform a confidenceguided optimization. Moreover, we introduce RoadSight, a multi-modality, multi-view dataset from real scenarios in infrastructure views. To our knowledge, I2V-GS is the first framework to generate autonomous driving datasets with infrastructure-vehicle view transformation. Experimental results demonstrate that I2V-GS significantly improves synthesis quality under vehicle view, outperforming StreetGaussian in NTA-Iou, NTL-Iou, and FID by 45.7%, 34.2%, and 14.9%, respectively.
  
<img width="1294" height="616" alt="image" src="https://github.com/user-attachments/assets/967a520a-1494-4996-85bd-9bf02e9c8402" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.23683) | [⌨️ Code] | [🌐 Project Page]

#### <summary>LT-Gaussian: Long-Term Map Update Using 3D Gaussian Splatting for Autonomous Driving
>*a map update method for 3D-GS-based maps*

Authors: Luqi Cheng, Zhangshuo Qi, Zijie Zhou, Chao Lu, Guangming Xiong
<details span>
<summary><b>Abstract</b></summary>
Maps play an important role in autonomous driving systems. The recently proposed 3D Gaussian Splatting (3D-GS) produces rendering-quality explicit scene reconstruction results, demonstrating the potential for map construction in autonomous driving scenarios. However, because of the time and computational costs involved in generating Gaussian scenes, how to update the map becomes a significant challenge. In this paper, we propose LT-Gaussian, a map update method for 3D-GS-based maps. LT-Gaussian consists of three main components: Multimodal Gaussian Splatting, Structural Change Detection Module, and Gaussian-Map Update Module. Firstly, the Gaussian map of the old scene is generated using our proposed Multimodal Gaussian Splatting. Subsequently, during the map update process, we compare the outdated Gaussian map with the current LiDAR data stream to identify structural changes. Finally, we perform targeted updates to the Gaussian-map to generate an up-to-date map. We establish a benchmark for map updating on the nuScenes dataset to quantitatively evaluate our method. The experimental results show that LT-Gaussian can effectively and efficiently update the Gaussian-map, handling common environmental changes in autonomous driving scenarios. Furthermore, by taking full advantage of information from both new and old scenes, LT-Gaussian is able to produce higher quality reconstruction results compared to map update strategies that reconstruct maps from scratch.
  
<img width="1934" height="626" alt="image" src="https://github.com/user-attachments/assets/bbacd9c0-9302-4472-a664-ad705e3b2faa" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.01704) | [⌨️ Code](https://github.com/ChengLuqi/LT-gaussian) | [🌐 Project Page]


#### <summary>VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling
>*Uncertainty Distillation from distractor-free NeRF approaches (Ren et al. 2024; Martin-Brualla et al. 2021)*

Authors: Yuru Xiao, Zihan Lin, Chao Lu, Deming Zhai, Kui Jiang, Wenbo Zhao, Wei Zhang, Junjun Jiang, Huanran Wang, Xianming Liu
<details span>
<summary><b>Abstract</b></summary>
Dynamic urban scene modeling is a rapidly evolving area with broad applications. While current approaches leveraging neural radiance fields or Gaussian Splatting have achieved fine-grained reconstruction and high-fidelity novel view synthesis, they still face significant limitations. These often stem from a dependence on pre-calibrated object tracks or difficulties in accurately modeling fast-moving objects from undersampled capture, particularly due to challenges in handling temporal discontinuities. To overcome these issues, we propose a novel video diffusion-enhanced 4D Gaussian Splatting framework. Our key insight is to distill robust, temporally consistent priors from a test-time adapted video diffusion model. To ensure precise pose alignment and effective integration of this denoised content, we introduce two core innovations: a joint timestamp optimization strategy that refines interpolated frame poses, and an uncertainty distillation method that adaptively extracts target content while preserving well-reconstructed regions. Extensive experiments demonstrate that our method significantly enhances dynamic modeling, especially for fast-moving objects, achieving an approximate PSNR gain of 2 dB for novel view synthesis over baseline approaches.
  
<img width="1706" height="746" alt="image" src="https://github.com/user-attachments/assets/607e79e4-c665-4cf1-941c-6fcd5e4ff080" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.02129) | [⌨️ Code](https://github.com/pulangk97/VDEGaussian) | [🌐 Project Page](https://pulangk97.github.io/VDEGaussian-Project/)


#### <summary>GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments
>*combines 3D Gaussian representation with continual learning for the first time*

Authors: Lin Zeng, Boming Zhao, Jiarui Hu, Xujie Shen, Ziqiang Dang, Hujun Bao, Zhaopeng Cui
<details span>
<summary><b>Abstract</b></summary>
Novel view synthesis with neural models has advanced rapidly in recent years, yet adapting these models to scene changes remains an open problem. Existing methods are either labor-intensive, requiring extensive model retraining, or fail to capture detailed types of changes over time. In this paper, we present GaussianUpdate, a novel approach that combines 3D Gaussian representation with continual learning to address these challenges. Our method effectively updates the Gaussian radiance fields with current data while preserving information from past scenes. Unlike existing methods, GaussianUpdate explicitly models different types of changes through a novel multi-stage update strategy. Additionally, we introduce a visibility-aware continual learning approach with generative replay, enabling self-aware updating without the need to store images. The experiments on the benchmark dataset demonstrate our method achieves superior and real-time rendering with the capability of visualizing changes over different times.

<img width="1580" height="902" alt="image" src="https://github.com/user-attachments/assets/151b0536-d753-4cfd-8607-b3bf0cd348d8" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2508.08867) | [⌨️ Code] | [🌐 Project Page]

#### <summary>InstDrive: Instance-Aware 3D Gaussian Splatting for Driving Scenes
>*segment for scene Decoupling*

Authors: Hongyuan Liu, Haochen Yu, Jianfei Jiang, Qiankun Liu, Jiansheng Chen, Huimin Ma
<details span>
<summary><b>Abstract</b></summary>
Reconstructing dynamic driving scenes from dashcam videos has attracted increasing attention due to its significance in autonomous driving and scene understanding. While recent advances have made impressive progress, most methods still unify all background elements into a single representation, hindering both instance-level understanding and flexible scene editing. Some approaches attempt to lift 2D segmentation into 3D space, but often rely on pre-processed instance IDs or complex pipelines to map continuous features to discrete identities. Moreover, these methods are typically designed for indoor scenes with rich viewpoints, making them less applicable to outdoor driving scenarios. In this paper, we present InstDrive, an instance-aware 3D Gaussian Splatting framework tailored for the interactive reconstruction of dynamic driving scene. We use masks generated by SAM as pseudo ground-truth to guide 2D feature learning via contrastive loss and pseudo-supervised objectives. At the 3D level, we introduce regularization to implicitly encode instance identities and enforce consistency through a voxel-based loss. A lightweight static codebook further bridges continuous features and discrete identities without requiring data pre-processing or complex optimization. Quantitative and qualitative experiments demonstrate the effectiveness of InstDrive, and to the best of our knowledge, it is the first framework to achieve 3D instance segmentation in dynamic, open-world driving.

<img width="1630" height="765" alt="image" src="https://github.com/user-attachments/assets/ea7eae67-d529-4b77-a80a-85f454cdb95f" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.12015) | [⌨️ Code] | [🌐 Project Page](https://instdrive.github.io/)

#### <summary>Distilled-3DGS:Distilled 3D Gaussian Splatting
>*propose a voxel histogram-based structural loss to enhance the structural learning capability of the student model*

Authors: Lintao Xiang, Xinkai Chen, Jianhuang Lai, Guangcong Wang
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noise-augmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGS, a simple yet effective framework without bells and whistles, achieves promising rendering results in both rendering quality and storage efficiency compared to state-of-the-art methods.

<img width="1564" height="810" alt="image" src="https://github.com/user-attachments/assets/49d5dabb-fe49-4832-b575-f16bc4ba3105" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.14037) | [⌨️ Code](https://github.com/lt-xiang/Distilled-3DGS) | [🌐 Project Page](https://distilled3dgs.github.io/)


#### <summary>Online 3D Gaussian Splatting Modeling with Novel View Selection
>*largest eigenvalue and positional gradients for Uncertainty Estimation*

Authors: Byeonggwon Lee, Junkyu Park, Khang Truong Giang, Soohwan Song
<details span>
<summary><b>Abstract</b></summary>
This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.

<img width="2121" height="725" alt="image" src="https://github.com/user-attachments/assets/5425d485-71f2-40a1-90d2-4c3e0e5ff929" />


</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.14014) | [⌨️ Code] | [🌐 Project Page]

#### <summary>DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians
>*near-to-far + depth and normal*

Authors: Cong Wang, Xianda Guo, Wenbo Xu, Wei Tian, Ruiqi Song, Chenming Zhang, Lingxi Li, Long Chen
<details span>
<summary><b>Abstract</b></summary>
In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.

<img width="1660" height="556" alt="image" src="https://github.com/user-attachments/assets/6ec9bb6b-70f8-46b0-8ab8-42218d79fc41" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.15376) | [⌨️ Code](https://github.com/Michael-Evans-Savitar/DriveSplat) | [🌐 Project Page](https://physwm.github.io/drivesplat/)

#### <summary>ExtraGS: Geometric-Aware Trajectory Extrapolation with Uncertainty-Guided Generative Priors
>*a new Gaussian node termed Far Field Gaussians GFFG that jointly modulates both the position and scale of each Gaussian primitive*

Authors: Kaiyuan Tan, Yingying Shen, Haohui Zhu, Zhiwei Zhan, Shan Zhao, Mingfei Tu, Hongcheng Luo, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye
<details span>
<summary><b>Abstract</b></summary>
Synthesizing extrapolated views from recorded driving logs is critical for simulating driving scenes for autonomous driving vehicles, yet it remains a challenging task. Recent methods leverage generative priors as pseudo ground truth, but often lead to poor geometric consistency and over-smoothed renderings. To address these limitations, we propose ExtraGS, a holistic framework for trajectory extrapolation that integrates both geometric and generative priors. At the core of ExtraGS is a novel Road Surface Gaussian(RSG) representation based on a hybrid Gaussian-Signed Distance Function (SDF) design, and Far Field Gaussians (FFG) that use learnable scaling factors to efficiently handle distant objects. Furthermore, we develop a self-supervised uncertainty estimation framework based on spherical harmonics that enables selective integration of generative priors only where extrapolation artifacts occur. Extensive experiments on multiple datasets, diverse multi-camera setups, and various generative priors demonstrate that ExtraGS significantly enhances the realism and geometric consistency of extrapolated views, while preserving high fidelity along the original trajectory.

<img width="1624" height="661" alt="image" src="https://github.com/user-attachments/assets/458550cd-9eda-4eef-9b27-d63dc3e62a16" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.15529) | [⌨️ Code] | [🌐 Project Page](https://wm-research.github.io/extrags/)


#### <summary>LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding
>*Mesh for scene by diffusion*

Authors: Julian Ost, Andrea Ramazzina, Amogh Joshi, Maximilian Bömer, Mario Bijelic, Felix Heide
<details span>
<summary><b>Abstract</b></summary>
Large-scale scene data is essential for training and testing in robot learning. Neural reconstruction methods have promised the capability of reconstructing large physically-grounded outdoor scenes from captured sensor data. However, these methods have baked-in static environments and only allow for limited scene control -- they are functionally constrained in scene and trajectory diversity by the captures from which they are reconstructed. In contrast, generating driving data with recent image or video diffusion models offers control, however, at the cost of geometry grounding and causality. In this work, we aim to bridge this gap and present a method that directly generates large-scale 3D driving scenes with accurate geometry, allowing for causal novel view synthesis with object permanence and explicit 3D geometry estimation. The proposed method combines the generation of a proxy geometry and environment representation with score distillation from learned 2D image priors. We find that this approach allows for high controllability, enabling the prompt-guided geometry and high-fidelity texture and structure that can be conditioned on map layouts -- producing realistic and geometrically consistent 3D generations of complex driving scenes.

<img width="1624" height="661" alt="image" src="https://github.com/user-attachments/assets/97db38e6-21ff-467f-8690-45acdb4c4868" />


</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.19204) | [⌨️ Code] | [🌐 Project Page](https://princeton-computational-imaging.github.io/LSD-3D/index.html)

#### <summary>DrivingGaussian++: Towards Realistic Reconstruction and Editable Simulation for Surrounding Dynamic Driving Scenes
>*scene editing*

Authors: Yajiao Xiong, Xiaoyu Zhou, Yongtao Wan, Deqing Sun, Ming-Hsuan Yang
<details span>
<summary><b>Abstract</b></summary>
We present DrivingGaussian++, an efficient and effective framework for realistic reconstructing and controllable editing of surrounding dynamic autonomous driving scenes. DrivingGaussian++ models the static background using incremental 3D Gaussians and reconstructs moving objects with a composite dynamic Gaussian graph, ensuring accurate positions and occlusions. By integrating a LiDAR prior, it achieves detailed and consistent scene reconstruction, outperforming existing methods in dynamic scene reconstruction and photorealistic surround-view synthesis. DrivingGaussian++ supports training-free controllable editing for dynamic driving scenes, including texture modification, weather simulation, and object manipulation, leveraging multi-view images and depth priors. By integrating large language models (LLMs) and controllable editing, our method can automatically generate dynamic object motion trajectories and enhance their realism during the optimization process. DrivingGaussian++ demonstrates consistent and realistic editing results and generates dynamic multi-view driving scenarios, while significantly enhancing scene diversity. 

<img width="1655" height="565" alt="image" src="https://github.com/user-attachments/assets/bf3871f2-9e9a-4051-bd6e-38148f7a6760" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.20965) | [⌨️ Code] | [🌐 Project Page](https://xiong-creator.github.io/DrivingGaussian_plus.github.io/)


#### <summary>Realistic and Controllable 3D Gaussian-Guided Object Editing for Driving Video Generation
>*scene editing for 3dgs*

Authors: Jiusi Li, Jackson Jiang, Jinyu Miao, Miao Long, Tuopu Wen, Peijin Jia, Shengxiang Liu, Chunlei Yu, Maolin Liu, Yuzhan Cai, Kun Jiang, Mengmeng Yang, Diange Yang
<details span>
<summary><b>Abstract</b></summary>
Corner cases are crucial for training and validating autonomous driving systems, yet collecting them from the real world is often costly and hazardous. Editing objects within captured sensor data offers an effective alternative for generating diverse scenarios, commonly achieved through 3D Gaussian Splatting or image generative models. However, these approaches often suffer from limited visual fidelity or imprecise pose control. To address these issues, we propose G^2Editor, a framework designed for photorealistic and precise object editing in driving videos. Our method leverages a 3D Gaussian representation of the edited object as a dense prior, injected into the denoising process to ensure accurate pose control and spatial consistency. A scene-level 3D bounding box layout is employed to reconstruct occluded areas of non-target objects. Furthermore, to guide the appearance details of the edited object, we incorporate hierarchical fine-grained features as additional conditions during generation. Experiments on the Waymo Open Dataset demonstrate that G^2Editor effectively supports object repositioning, insertion, and deletion within a unified framework, outperforming existing methods in both pose controllability and visual quality, while also benefiting downstream data-driven tasks.

<img width="1405" height="569" alt="image" src="https://github.com/user-attachments/assets/59332bbc-a9f2-4b88-95c9-eb3ba661aeb2" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.20471) | [⌨️ Code] | [🌐 Project Page]


#### <summary>VGD: Visual Geometry Gaussian Splatting for Feed-Forward Surround-view Driving Reconstruction
>*DPT-GS*

Authors: Junhong Lin, Kangli Wang, Shunzhou Wang, Songlin Fan, Ge Li, Wei Gao
<details span>
<summary><b>Abstract</b></summary>
Feed-forward surround-view autonomous driving scene reconstruction offers fast, generalizable inference ability, which faces the core challenge of ensuring generalization while elevating novel view quality. Due to the surround-view with minimal overlap regions, existing methods typically fail to ensure geometric consistency and reconstruction quality for novel views. To tackle this tension, we claim that geometric information must be learned explicitly, and the resulting features should be leveraged to guide the elevating of semantic quality in novel views. In this paper, we introduce \textbf{Visual Gaussian Driving (VGD)}, a novel feed-forward end-to-end learning framework designed to address this challenge. To achieve generalizable geometric estimation, we design a lightweight variant of the VGGT architecture to efficiently distill its geometric priors from the pre-trained VGGT to the geometry branch. Furthermore, we design a Gaussian Head that fuses multi-scale geometry tokens to predict Gaussian parameters for novel view rendering, which shares the same patch backbone as the geometry branch. Finally, we integrate multi-scale features from both geometry and Gaussian head branches to jointly supervise a semantic refinement model, optimizing rendering quality through feature-consistent learning. Experiments on nuScenes demonstrate that our approach significantly outperforms state-of-the-art methods in both objective metrics and subjective quality under various settings, which validates VGD's scalability and high-fidelity surround-view reconstruction.

<img width="2124" height="741" alt="image" src="https://github.com/user-attachments/assets/d3f0efda-8b2e-475e-bffc-708b69d7b945" />

</details>

[📃 arXiv:2510](https://arxiv.org/pdf/2510.19578) | [⌨️ Code](https://github.com/JHLin42in/VGD) | [🌐 Project Page]



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


#### <summary>SAM4D: Segment Anything in Camera and LiDAR Streams
> *lidar segmentation*

Authors: Jianyun Xu, Song Wang, Ziqian Ni, Chunyong Hu, Sheng Yang, Jianke Zhu, Qiang Li
<details span>
<summary><b>Abstract</b></summary>
We present SAM4D, a multi-modal and temporal foundation model designed for promptable segmentation across camera and LiDAR streams. Unified Multi-modal Positional Encoding (UMPE) is introduced to align camera and LiDAR features in a shared 3D space, enabling seamless cross-modal prompting and interaction. Additionally, we propose Motion-aware Cross-modal Memory Attention (MCMA), which leverages ego-motion compensation to enhance temporal consistency and long-horizon feature retrieval, ensuring robust segmentation across dynamically changing autonomous driving scenes. To avoid annotation bottlenecks, we develop a multi-modal automated data engine that synergizes VFM-driven video masklets, spatiotemporal 4D reconstruction, and cross-modal masklet fusion. This framework generates camera-LiDAR aligned pseudo-labels at a speed orders of magnitude faster than human annotation while preserving VFM-derived semantic fidelity in point cloud representations. We conduct extensive experiments on the constructed Waymo-4DSeg, which demonstrate the powerful cross-modal segmentation ability and great potential in data annotation of proposed SAM4D.

<img width="1992" height="822" alt="image" src="https://github.com/user-attachments/assets/98e5dd4e-1141-421c-9b09-0935242c859d" />

</details>

[📃 arXiv:2506](https://arxiv.org/pdf/2506.21547) | [⌨️ Code](https://github.com/CN-ADLab/SAM4D) | [🌐 Project Page](https://sam4d-project.github.io/)



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

#### <summary>RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration
>*measuring similarity between the GMMs through an optimal transport MW2 distance by an entropy-regularized Sinkhorn approach*

Authors: Chong Cheng, Yu Hu, Sicheng Yu, Beizhen Zhao, Zijian Wang, Hao Wang
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has demonstrated its potential in reconstructing scenes from unposed images. However, optimization-based 3DGS methods struggle with sparse views due to limited prior knowledge. Meanwhile, feed-forward Gaussian approaches are constrained by input formats, making it challenging to incorporate more input views. To address these challenges, we propose RegGS, a 3D Gaussian registration-based framework for reconstructing unposed sparse views. RegGS aligns local 3D Gaussians generated by a feed-forward network into a globally consistent 3D Gaussian representation. Technically, we implement an entropy-regularized Sinkhorn algorithm to efficiently solve the optimal transport Mixture 2-Wasserstein  distance, which serves as an alignment metric for Gaussian mixture models (GMMs) in  space. Furthermore, we design a joint 3DGS registration module that integrates the  distance, photometric consistency, and depth geometry. This enables a coarse-to-fine registration process while accurately estimating camera poses and aligning the scene. Experiments on the RE10K and ACID datasets demonstrate that RegGS effectively registers local Gaussians with high fidelity, achieving precise pose estimation and high-quality novel-view synthesis.

![image](https://github.com/user-attachments/assets/c2a3043f-05bf-43d8-a46b-c9f6a7d49c05)

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.08136) | [⌨️ Code] | [🌐 Project Page]


#### <summary>Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics
>*Can trajectory prediction be approached from a planning perspective and enhanced with intention reasoning capabilities?*

Authors: Muleilan Pei, Shaoshuai Shi, Xuesong Chen, Xu Liu, Shaojie Shen
<details span>
<summary><b>Abstract</b></summary>
Motion forecasting for on-road traffic agents presents both a significant challenge and a critical necessity for ensuring safety in autonomous driving systems. In contrast to most existing data-driven approaches that directly predict future trajectories, we rethink this task from a planning perspective, advocating a "First Reasoning, Then Forecasting" strategy that explicitly incorporates behavior intentions as spatial guidance for trajectory prediction. To achieve this, we introduce an interpretable, reward-driven intention reasoner grounded in a novel query-centric Inverse Reinforcement Learning (IRL) scheme. Our method first encodes traffic agents and scene elements into a unified vectorized representation, then aggregates contextual features through a query-centric paradigm. This enables the derivation of a reward distribution, a compact yet informative representation of the target agent's behavior within the given scene context via IRL. Guided by this reward heuristic, we perform policy rollouts to reason about multiple plausible intentions, providing valuable priors for subsequent trajectory generation. Finally, we develop a hierarchical DETR-like decoder integrated with bidirectional selective state space models to produce accurate future trajectories along with their associated probabilities. Extensive experiments on the large-scale Argoverse and nuScenes motion forecasting datasets demonstrate that our approach significantly enhances trajectory prediction confidence, achieving highly competitive performance relative to state-of-the-art methods.

<img width="1614" height="899" alt="image" src="https://github.com/user-attachments/assets/86fb69db-aac4-4754-8157-efc2c5f1e054" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.12083) | [⌨️ Code] | [🌐 Project Page]


#### <summary>ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting
>*three kinds of voting strategies to quickly initialize the point cloud for different objects: (1) Majority Voting. (2) Probability-based Voting. (3)Correspondence-based Voting.*

Authors: Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, Bo Dai
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing.

<img width="1785" height="711" alt="image" src="https://github.com/user-attachments/assets/2f536957-c827-4682-a6be-c36c6c787a61" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.15454) | [⌨️ Code](https://github.com/RuijieZhu94/ObjectGS) | [🌐 Project Page](https://ruijiezhu94.github.io/ObjectGS_page/)



#### <summary>Gaussian Set Surface Reconstruction through Per-Gaussian Optimization
>*enforce single-view normal consistency and multi-view photometric consistency on each Gaussian instance*

Authors: Zhentao Huang, Di Wu, Zhenbang He, Minglun Gong
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) effectively synthesizes novel views through its flexible representation, yet fails to accurately reconstruct scene geometry. While modern variants like PGSR introduce additional losses to ensure proper depth and normal maps through Gaussian fusion, they still neglect individual placement optimization. This results in unevenly distributed Gaussians that deviate from the latent surface, complicating both reconstruction refinement and scene editing. Motivated by pioneering work on Point Set Surfaces, we propose Gaussian Set Surface Reconstruction (GSSR), a method designed to distribute Gaussians evenly along the latent surface while aligning their dominant normals with the surface normal. GSSR enforces fine-grained geometric alignment through a combination of pixel-level and Gaussian-level single-view normal consistency and multi-view photometric consistency, optimizing both local and global perspectives. To further refine the representation, we introduce an opacity regularization loss to eliminate redundant Gaussians and apply periodic depth- and normal-guided Gaussian reinitialization for a cleaner, more uniform spatial distribution. Our reconstruction results demonstrate significantly improved geometric precision in Gaussian placement, enabling intuitive scene editing and efficient generation of novel Gaussian-based 3D environments. Extensive experiments validate GSSR's effectiveness, showing enhanced geometric accuracy while preserving high-quality rendering performance.

<img width="970" height="854" alt="image" src="https://github.com/user-attachments/assets/52b12e57-8eab-4761-849b-06c7b8cd4d13" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.18923) | [⌨️ Code] | [🌐 Project Page]



#### <summary>CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception
>*two MLPs are utilized to predict the parameters of the latent rotation and the latent translation, respectively*

Authors: Jiaru Zhong, Jiahao Wang, Jiahui Xu, Xiaofan Li, Zaiqing Nie, Haibao Yu
<details span>
<summary><b>Abstract</b></summary>
Cooperative perception aims to address the inherent limitations of single-vehicle autonomous driving systems through information exchange among multiple agents. Previous research has primarily focused on single-frame perception tasks. However, the more challenging cooperative sequential perception tasks, such as cooperative 3D multi-object tracking, have not been thoroughly investigated. Therefore, we propose CoopTrack, a fully instance-level end-to-end framework for cooperative tracking, featuring learnable instance association, which fundamentally differs from existing approaches. CoopTrack transmits sparse instance-level features that significantly enhance perception capabilities while maintaining low transmission costs. Furthermore, the framework comprises two key components: Multi-Dimensional Feature Extraction, and Cross-Agent Association and Aggregation, which collectively enable comprehensive instance representation with semantic and motion features, and adaptive cross-agent association and fusion based on a feature graph. Experiments on both the V2X-Seq and Griffin datasets demonstrate that CoopTrack achieves excellent performance. Specifically, it attains state-of-the-art results on V2X-Seq, with 39.0\% mAP and 32.8\% AMOTA.

<img width="922" height="783" alt="image" src="https://github.com/user-attachments/assets/c4145046-33c6-4c12-8352-61e073dd1031" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.19239) | [⌨️ Code](https://github.com/zhongjiaru/CoopTrack) | [🌐 Project Page]



#### <summary>Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction
> *the clone operation is mainly responsible for local refinement, while the split operation takes charge of the global diffusion. *

Authors: Binxiao Huang, Zhengwu Liu, Ngai Wong
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarse-to-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending, and Tanks & Temples datasets demonstrate that our approach significantly accelerates training,achieving over 2x speedup with fewer Gaussian primitives and superior reconstruction performance.

<img width="1848" height="547" alt="image" src="https://github.com/user-attachments/assets/45b1065a-8f95-47a5-acfc-c30c1eca8d2f" />

</details>

[📃 arXiv:2507](https://arxiv.org/pdf/2507.20239) | [⌨️ Code] | [🌐 Project Page]



#### <summary>Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians
> *diffusion for denoising gaussians*

Authors: Quankai Gao, Iliyan Georgiev, Tuanfeng Y. Wang, Krishna Kumar Singh, Ulrich Neumann, Jae Shin Yoon
<details span>
<summary><b>Abstract</b></summary>
3D generation has made significant progress, however, it still largely remains at the object-level. Feedforward 3D scene-level generation has been rarely explored due to the lack of models capable of scaling-up latent representation learning on 3D scene-level data. Unlike object-level generative models, which are trained on well-labeled 3D data in a bounded canonical space, scene-level generations with 3D scenes represented by 3D Gaussian Splatting (3DGS) are unbounded and exhibit scale inconsistency across different scenes, making unified latent representation learning for generative purposes extremely challenging. In this paper, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which effectively captures both semantic and spatial information of the inputs. Beyond model design, we propose a general pipeline for 3D scene data processing to address scale inconsistency issue. We validate our method on the recent scene-level 3D dataset DL3DV-10K, where we found that only Can3Tok successfully generalizes to novel 3D scenes, while compared methods fail to converge on even a few hundred scene inputs during training and exhibit zero generalization ability during inference. Finally, we demonstrate image-to-3DGS and text-to-3DGS generation as our applications to demonstrate its ability to facilitate downstream generation tasks.

<img width="2086" height="532" alt="image" src="https://github.com/user-attachments/assets/af5c1715-c33e-4836-b89e-8d729eb0f7a5" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.01464) | [⌨️ Code](https://github.com/Zerg-Overmind/Can3Tok) | [🌐 Project Page](https://zerg-overmind.github.io/Can3Tok.github.io/)

#### <summary>TopoImages: Incorporating Local Topology Encoding into Deep Learning Models for Medical Image Classification
> *persistence diagrams*

Authors: Pengfei Gu, Hongxiao Wang, Yejia Zhang, Huimin Li, Chaoli Wang, Danny Chen
<details span>
<summary><b>Abstract</b></summary>
Topological structures in image data, such as connected components and loops, play a crucial role in understanding image content (e.g., biomedical objects). % Despite remarkable successes of numerous image processing methods that rely on appearance information, these methods often lack sensitivity to topological structures when used in general deep learning (DL) frameworks. % In this paper, we introduce a new general approach, called TopoImages (for Topology Images), which computes a new representation of input images by encoding local topology of patches. % In TopoImages, we leverage persistent homology (PH) to encode geometric and topological features inherent in image patches. % Our main objective is to capture topological information in local patches of an input image into a vectorized form. % Specifically, we first compute persistence diagrams (PDs) of the patches, % and then vectorize and arrange these PDs into long vectors for pixels of the patches. % The resulting multi-channel image-form representation is called a TopoImage. % TopoImages offers a new perspective for data analysis. % To garner diverse and significant topological features in image data and ensure a more comprehensive and enriched representation, we further generate multiple TopoImages of the input image using various filtration functions, which we call multi-view TopoImages. % The multi-view TopoImages are fused with the input image for DL-based classification, with considerable improvement. % Our TopoImages approach is highly versatile and can be seamlessly integrated into common DL frameworks. Experiments on three public medical image classification datasets demonstrate noticeably improved accuracy over state-of-the-art methods.

<img width="1712" height="352" alt="image" src="https://github.com/user-attachments/assets/355112f2-ee48-459f-a34f-c2de578a4213" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.01574) | [⌨️ Code] | [🌐 Project Page]


#### <summary>Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey

Authors: Yan Gong, Naibang Wang, Jianli Lu, Xinyu Zhang, Yongsheng Gao, Jie Zhao, Zifan Huang, Haozhi Bai, Nanxin Zeng, Nayu Su, Lei Yang, Ziying Song, Xiaoxi Hu, Xinmin Jiang, Xiaojuan Zhang, Susanto Rahardja
<details span>
<summary><b>Abstract</b></summary>
Bird's-Eye-View (BEV) perception has become a foundational paradigm in autonomous driving, enabling unified spatial representations that support robust multi-sensor fusion and multi-agent collaboration. As autonomous vehicles transition from controlled environments to real-world deployment, ensuring the safety and reliability of BEV perception in complex scenarios - such as occlusions, adverse weather, and dynamic traffic - remains a critical challenge. This survey provides the first comprehensive review of BEV perception from a safety-critical perspective, systematically analyzing state-of-the-art frameworks and implementation strategies across three progressive stages: single-modality vehicle-side, multimodal vehicle-side, and multi-agent collaborative perception. Furthermore, we examine public datasets encompassing vehicle-side, roadside, and collaborative settings, evaluating their relevance to safety and robustness. We also identify key open-world challenges - including open-set recognition, large-scale unlabeled data, sensor degradation, and inter-agent communication latency - and outline future research directions, such as integration with end-to-end autonomous driving systems, embodied intelligence, and large language models.

<img width="1168" height="960" alt="image" src="https://github.com/user-attachments/assets/d8aa50ad-73b1-4c15-a064-6f1f5c65ce7c" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.07560) | [⌨️ Code] | [🌐 Project Page]

#### <summary>GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting

Authors: Jiajun Dong, Chengkun Wang, Wenzhao Zheng, Lei Chen, Jiwen Lu, Yansong Tang
<details span>
<summary><b>Abstract</b></summary>
Effective image tokenization is crucial for both multi-modal understanding and generation tasks due to the necessity of the alignment with discrete text data. To this end, existing approaches utilize vector quantization (VQ) to project pixels onto a discrete codebook and reconstruct images from the discrete representation. However, compared with the continuous latent space, the limited discrete codebook space significantly restrict the representational ability of these image tokenizers. In this paper, we propose GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting as a solution. We first represent the encoded samples as multiple flexible featured 2D Gaussians characterized by positions, rotation angles, scaling factors, and feature coefficients. We adopt the standard quantization for the Gaussian features and then concatenate the quantization results with the other intrinsic Gaussian parameters before the corresponding splatting operation and the subsequent decoding module. In general, GaussianToken integrates the local influence of 2D Gaussian distribution into the discrete space and thus enhances the representation capability of the image tokenizer. Competitive reconstruction performances on CIFAR, Mini-ImageNet, and ImageNet-1K demonstrate the effectiveness of our framework.

<img width="1502" height="683" alt="image" src="https://github.com/user-attachments/assets/0f8d186e-8a42-4caf-b060-6fc2eef37103" />

</details>

[📃 arXiv:2501](https://arxiv.org/pdf/2501.15619v1) | [⌨️ Code](https://github.com/ChrisDong-THU/GaussianToken) | [🌐 Project Page]


#### <summary>ROVR-Open-Dataset: A Large-Scale Depth Dataset for Autonomous Driving

Authors: Xianda Guo, Ruijun Zhang, Yiqun Duan, Ruilin Wang, Keyuan Zhou, Wenzhao Zheng, Wenke Huang, Gangwei Xu, Mike Horton, Yuan Si, Hao Zhao, Long Chen
<details span>
<summary><b>Abstract</b></summary>
Depth estimation is a fundamental task for 3D scene understanding in autonomous driving, robotics, and augmented reality. Existing depth datasets, such as KITTI, nuScenes, and DDAD, have advanced the field but suffer from limitations in diversity and scalability. As benchmark performance on these datasets approaches saturation, there is an increasing need for a new generation of large-scale, diverse, and cost-efficient datasets to support the era of foundation models and multi-modal learning. To address these challenges, we introduce a large-scale, diverse, frame-wise continuous dataset for depth estimation in dynamic outdoor driving environments, comprising 20K video frames to evaluate existing methods. Our lightweight acquisition pipeline ensures broad scene coverage at low cost, while sparse yet statistically sufficient ground truth enables robust training. Compared to existing datasets, ours presents greater diversity in driving scenarios and lower depth density, creating new challenges for generalization. Benchmark experiments with standard monocular depth estimation models validate the dataset's utility and highlight substantial performance gaps in challenging conditions, establishing a new platform for advancing depth estimation research.

<img width="1770" height="999" alt="image" src="https://github.com/user-attachments/assets/21da186e-3ecd-4821-a628-d8c36c4537f9" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.13977) | [⌨️ Code](https://github.com/rovr-network/ROVR-Open-Dataset) | [🌐 Project Page](https://xiandaguo.net/ROVR-Open-Dataset/)


#### <summary>CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes
>*As evidenced in the left portion of Fig. 3, these tiny, sand-like projected points contribute substantially to points with high gradients. And they belong to those with extreme elongation. Moreover, some points project smaller than one pixel, resulting in their covariance being replaced by a fixed value through the antialiased low-pass filter. Consequently, these points cannot properly adjust their scaling and rotation with valid gradients.*

Authors: Yang Liu, Chuanchen Luo, Zhongkai Mao, Junran Peng, Zhaoxiang Zhang
<details span>
<summary><b>Abstract</b></summary>
Recently, 3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction, manifesting efficient and high-fidelity novel view synthesis. However, accurately representing surfaces, especially in large and complex scenarios, remains a significant challenge due to the unstructured nature of 3DGS. In this paper, we present CityGaussianV2, a novel approach for large-scale scene reconstruction that addresses critical challenges related to geometric accuracy and efficiency. Building on the favorable generalization capabilities of 2D Gaussian Splatting (2DGS), we address its convergence and scalability issues. Specifically, we implement a decomposed-gradient-based densification and depth regression technique to eliminate blurry artifacts and accelerate convergence. To scale up, we introduce an elongation filter that mitigates Gaussian count explosion caused by 2DGS degeneration. Furthermore, we optimize the CityGaussian pipeline for parallel training, achieving up to 10 compression, at least 25% savings in training time, and a 50% decrease in memory usage. We also established standard geometry benchmarks under large-scale scenes. Experimental results demonstrate that our method strikes a promising balance between visual quality, geometric accuracy, as well as storage and training costs. 

<img width="1707" height="904" alt="image" src="https://github.com/user-attachments/assets/570fb353-bd02-44ef-a6f0-66ed9d6c1bbc" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2411.00771) | [⌨️ Code](https://github.com/Linketic/CityGaussian) | [🌐 Project Page](https://dekuliutesla.github.io/CityGaussianV2/)


#### <summary>MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting
>*Weighted Chamfer Distance Loss*

Authors: Hanzhi Chang, Ruijie Zhu, Wenjie Chang, Mulin Yu, Yanzhe Liang, Jiahao Lu, Zhuoyuan Li, Tianzhu Zhang
<details span>
<summary><b>Abstract</b></summary>
Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks.

<img width="1540" height="675" alt="image" src="https://github.com/user-attachments/assets/0b06f73d-0b3c-439d-9d18-d68fb6146828" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.17811) | [⌨️ Code](https://github.com/HanzhiChang/MeshSplat) | [🌐 Project Page](https://hanzhichang.github.io/meshsplat_web/)

#### <summary>Complete Gaussian Splats from a Single Image with Denoising Diffusion Models
> *diffusion for Splatter Images*

Authors: Ziwei Liao, Mohamed Sayed, Steven L. Waslander, Sara Vicente, Daniyar Turmukhambetov, Michael Firman
<details span>
<summary><b>Abstract</b></summary>
Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings.

<img width="1886" height="419" alt="image" src="https://github.com/user-attachments/assets/a50a8770-c2a1-4f8c-913d-82f80c4a0920" />

</details>

[📃 arXiv:2508](https://arxiv.org/pdf/2508.21542) | [⌨️ Code] | [🌐 Project Page](https://nianticspatial.github.io/completesplat/)


#### <summary>Scaling Transformer-Based Novel View Synthesis Models with Token Disentanglement and Synthetic Data
> *Token-Disentangled (Tok-D) Transformer: In LVSM, the transformer blocks process source and target tokens in the same manner, even though the source consists of images and Pl¨ ucker rays, while the target includes only Pl¨ucker rays. Additionally, source and target image quality can differ when training with synthetic data.*

Authors: Nithin Gopalakrishnan Nair, Srinivas Kaza, Xuan Luo, Vishal M. Patel, Stephen Lombardi, Jungyeon Park
<details span>
<summary><b>Abstract</b></summary>
Large transformer-based models have made significant progress in generalizable novel view synthesis (NVS) from sparse input views, generating novel viewpoints without the need for test-time optimization. However, these models are constrained by the limited diversity of publicly available scene datasets, making most real-world (in-the-wild) scenes out-of-distribution. To overcome this, we incorporate synthetic training data generated from diffusion models, which improves generalization across unseen domains. While synthetic data offers scalability, we identify artifacts introduced during data generation as a key bottleneck affecting reconstruction quality. To address this, we propose a token disentanglement process within the transformer architecture, enhancing feature separation and ensuring more effective learning. This refinement not only improves reconstruction quality over standard transformers but also enables scalable training with synthetic data. As a result, our method outperforms existing models on both in-dataset and cross-dataset evaluations, achieving state-of-the-art results across multiple benchmarks while significantly reducing computational costs.

<img width="1626" height="693" alt="image" src="https://github.com/user-attachments/assets/0fd8581d-8f5b-4d38-bc75-6451388f5606" />

</details>

[📃 arXiv:2509](https://arxiv.org/pdf/2509.06950) | [⌨️ Code] | [🌐 Project Page](https://scaling3dnvs.github.io/)

#### <summary>DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning
> *This SDF-to-opacity mapping is a differentiable function that effectively locks each Gaussian to the surface*

Authors: Wenzhi Guo, Bing Wang
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.

<img width="1286" height="674" alt="image" src="https://github.com/user-attachments/assets/b5ac1a58-edb4-462b-880d-4ee29f4a0fd1" />


</details>

[📃 arXiv:2509](https://arxiv.org/pdf/2509.07493) | [⌨️ Code](https://github.com/DARYL-GWZ/DIGS) | [🌐 Project Page]




