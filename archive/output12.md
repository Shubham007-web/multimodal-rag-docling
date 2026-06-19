## Introduction and Overview

GenAl applications and the unique challenges of designing such systems. It's also intended to serve as a guide for those who want to understand how GenAl is applied in practical scenarios.

This chapter explores two key topics. First, it provides an overview of GenAl, delving into its fundamental concepts and applications. Then, it introduces a comprehensive framework for building ML systems, which is essential for real-world applications and interview preparation. This framework will serve as the foundation for developing popular GenAl systems in the following chapters.

Let's dive in.

## GenAl overview

Artificial intelligence(Al) is a branch of computer science focused on creating systems that can perform tasks that typically require human intelligence, such as reasoning, planning, and problem-solving.ML is a subsetof Al that uses algorithms to learn from data rather than relying on predefined rules. These algorithms analyze data, identify patterns, and make predictions or generate new content based on the learned patterns. Applications such as recommendation systems,fraud detection,autonomous vehicles,and chatbots are generally powered by ML models.

ML models generally fall into two categories:

- ·Discriminative
- ·Generative

## Discriminative

Discriminative models classify data by learning the differences between classes based on input features. Formally, they learn the conditional probabilities, P(Y | X), where Y represents the target variable and X represents the input features.

Discriminative models can be used forboth classification,where the goal is to determine the class to which an input belongs, and regression, where the goal is to predict a continuous value. For example,in fraud detection,a discriminative model might classify transactions as either legitimate or fraudulent by analyzing features such as transaction amount and purchase history. Similarly, in movie recommendations, a model predicts a user's rating for a movie based on the user's historical interactions.

Common algorithms for developing discriminative models include:

- ·Logistic regression: A linear model that predicts the probability of a binary outcome based on input features.
- Support vector machines (SVMs): SVMs find the hyperplanes that best separate classes in the feature space. They can be extended to learn non-linear boundaries using kernel functions [2].
- ·Decision trees: These models and their variations, such as random forests, recursively split the data into subgroups based on the target variable.
- on the majority label among its nearest neighbors in the feature space.
- complex functions for tasks such as classification and regression.

Figure 1.1: The relationship between Al and ML

<!-- image -->

instances.For that,we turn to generative models.

## Generativemodels

Formally, they model the distribution P(X) when focusing solely on the input data (e.g. image generation), or the joint probability distribution P(X,Y) when considering both the input data and the target variable (e.g., text-to-image generation). This allows them to generate new data instances by sampling from these learned distributions.

Unlike discriminative models,which focus on distinguishing data instances, generative models can create new data samples that closely resemble the original data.For instance, These models are applied in various tasks such as text generation, image generation, and speech synthesis.

Generative algorithms can be divided into two categories: classical and modern.Classical algorithms are good at learning patterns from structured data. However, they can struggle to learn from more complex or unstructured data. Common classical generative algorithms include:

- ·Naive Bayes: A probabilistic model based on Bayes' theorem [3].
- ·Gaussian mixture models (GMMs): GMMs [4] represent data as a mixture of Gaussian distributions.
- Hidden Markov models (HMMs): HMMs [5] model the joint probability of observed sequences and the hidden states generating those sequences.
- ·Boltzmann machines:Energy-based models used for feature learning or dimensionality reduction [6].

On the other hand,modern generative algorithms learn from complex data distributions and are well suited for tasks such as generating realistic images and producing accurate textual outputs in response to queries. Common modern generative algorithms include:

- Variational autoencoders(VAEs):A type of autoencoder that models the distribution of data by encoding it to a latent space and then reconstructing the original data using a decoder.
- Generative adversarial networks (GANs): A class of neural networks in which a generator and discriminator are trained simultaneously.The generator creates realistic data,and the discriminator tries to distinguish between real and generated data.
- Diffusion models: Models that learn complex data distributions through a reverse diffusion process. They are commonly used for image and video generation.
- ·Autoregressive models: Models that generate data by predicting each element in a se-

and time series forecasting.

and discriminative models.

Figure 1.2:Popular ML-powered tasks

<!-- image -->

## What is GenAl and why is it gaining popularity?

GenAl involves using modern generative algorithms to train models capable of producing new data samples such asimages,videos,text,and audio.

GenAl has gained a lot of popularity recently for two main reasons. First, these models can through 2040.

## WhyGenAl isbecoming so powerful?

powerful. Threekey factors driving this advancement are:

- 1.Data
- 2.Model capacity
- 3.Compute

An ML model's effectiveness depends on its training data.For example,if a model has not been trained on extensive medical data,it might struggle to diagnose diseases accurately. Improving a model for a specific task requires large datasets with labels, but collecting this data can be challenging and expensive.

One key driver behind the success of GenAl is self-supervised learning. Unlike classical models that typically work well when trained on labeled data, GenAl models can learn from unlabeled data.This approach lets them use vast datasets from theinternet without the need for costly and time-consuming labeling processes.

Figure 1.3:Data scale comparison between training a chatbot and disease diagnosis

<!-- image -->

Because of easy access to very large datasets from the internet, modern GenAl models can be trained on massive datasets,sometimes exceeding billions of text documents or images.For example,Meta's Llama 3 model [9] was trained on 15 trillion tokens-roughly 50 terabytes of data;Google's Flamingo model [10] was trained on 1.8 billion (image,text) pairs. Being trained on this massive amount of data helps these models learn complex

## Model capacity

capacity is measured in two ways:

- .Numberofparameters
- FLOPcount

## Numberofparameters

number of parameters is a key indicator of a model's capacity to learn from data.

A model with more parameters generally has a greater capacity to learn the complex patterns and relationships that exist within the data. This often translates to better perforay smus I't algel rasep aiel e uo paun uaaq sey apo ay usnsse ue popular models and their number of parameters.

Table 1.1: Popular GenAl models and their number of parameters

| Model Name             | Parameters   |
|------------------------|--------------|
| Google's PaLM [11]     | 540B         |
| OpenAl's GPT-3[12]     | 175B         |
| Google's Flamingo [10] | 80B          |
| Meta's Llama 3 [9]     | 405B         |
| Google's lmagen [13]   | 2B           |

## FLOPcount

FLOP (Floating Point Operations) measures the computational complexity of a model by movesthrough themodel'slayers.

To better understand FLOp count, let's walk through a simple example.

Figure 1.4: A simple fully connected layer and arithmetic calculations for a single output neuron

<!-- image -->

Consider a fully connected layer with 4 input neurons and 3 output neurons. Each output neuron is computed by multiplying the input neurons with their corresponding weight and summing them up. This results in 4 multiplications and 3 additions for each output neuron, as shown in Figure 1.4.Therefore,total FLOPs are 3×(4 + 3) = 21.

While the number ofparameters measures a model's size,FLoP indicates the number of arithmetic computations and provides insight into the model's computational complexity. Although a model with more parameters often has a higher FLOP count, this isn't always the case. The architecture plays a crucial role.For example, dense layers typically require more FLoPs than sparse connections,even if the parameter count is the same.Understanding this distinction is crucial when optimizing a model, as it helps us design models that are both accurate and computationally efficient.

## Compute

As models increase in capacity, their performance tends to improve, but training these large models requires enormous amounts of computational resources. The compute required during model training is often measured in FLOP,representing the total number of operations performed. For example, Google's PaLM-2 model was trained using 1022 FLOPs[14].,

Compute power is typically provided by hardware like CPUs, GPUs (Graphics Processing Units),and TPUs (Tensor Processing Units). Nvidia, for instance,offers advanced GPUs such as the H100,A100,and A10,each with different costs and processing capabilities. The performance of these machines is often measured in FLoP/S Floating Point Operations Per Second),For example,Nvidia'sH100 can deliver up to 60 teraflops per second (60TFLOP/S)[15].

stated that the cost of training GPT-4 was more than $100 million [16].

techniques have made it possible to train GenAl models at unprecedented scales.

## Scaling law

Within the constraints of a compute budget (measured in FLOPs), what is the optimal combination of model size and training data (measured by the number of tokens) that yields the lowest loss?This is the fundamental question researchers aim to answer through scaling laws.

In 2020, OpenAl researchers conducted extensive LLM training experiments, exploring various factors such as model sizes (N), dataset sizes (D), computational resources (C) model architectures, and context lengths [17]. Their findings revealed two key insights First, the impact of scaling on model performance is significantly more pronounced than the influence of architectural variations. Second, as model size, dataset size, or computaperformance,which follows a power-law trend.

Figure 1.5:OpenAl's scaling law (Credit:[17])

<!-- image -->

linearly with model size to achieve optimal performance.

about the existence of a scaling law during inference,as well[20].

## GenAl risks and limitations

GenAl has evolved quickly, driving advancements across many industries by creating realistic text, mages, and videos. However, it also brings critical risks and limitations that need careful consideration. Addressing these issues is key to ensuring responsible and sustainable development. Common challenges include:

- Ethical concerns:Issuesrelating to bias,intellectual property(Ip),misinformation,and misuse ofgenerated contentcan haveharmful societal impacts.
- contributes to substantial energy consumption and carbon emissions.
- Security risks: There are threats posed by the use of GenAl to create deepfakes used for blackmail, political manipulation, automated phishing attacks, and adversarial exploits that manipulate model outputs in critical systems such as healthcare and finance.
- ·Model limitations: GenAl models may lack true understanding,leading to inaccuracies and limitations in complexreasoning tasks,and hallucination.

Each of these areas represents a significant challenge in the development of GenAl applications. Addressing these risks requires a multidisciplinary approach involving not just technical solutionsbut also ethical frameworks,legal regulations,and societal awareness.

## A framework for ML system design interviews

Many engineers think of ML algorithms-for instance, autoregressive Transformers or diffusion models-as the entirety of an ML system. However, building and deploying GenAl systems involves much more than just training a model. These systems are complex, with components such as data pipelines to handle and preprocess large datasets,evaluation mechanisms to assess output quality and safety, infrastructure to deliver Al-generated content at scale,and monitoring to ensure consistent performance over time.

In an ML system design interview,particularly those focused on GenAl,you'll often face open-ended questions.For example,you might be asked to design a chatbot for customer service or an Al-powered image-editing tool for creatives. There isn't a single"correct" answer,The interviewer is interested in how you approach complex problems,your understanding of GenAl concepts,your system design process, and the reasoning behind your design choices.

framework includes the following key steps:

- 1.Clarifyingrequirements
- 2.FramingtheproblemasanMLtask
- 3.Data preparation
- 4.Modeldevelopment
- 5.Evaluation
- 6.Overall MLsystem design
7. Deployment and monitoring

Figure 1.6:ML system design steps

<!-- image -->

Let's dive into each step to examine the key considerations and talking points when designing a GenAl system.

## Clarifying Requirements

When you start to develop an ML system to solve a particular task, you often have minimal information to begin with. Similarly, in interviews, ML system design questions are often vague, providing minimal details. For instance, an interview might ask you to "design an image generation system." The first step is to ask clarifying questions. But what questions should you ask?

Your questions should help in understanding the problem space and the specifc goals the system needs to achieve.They fall into two types:

- ·Functional requirements
- ·Non-functionalrequirements

## Functional requirements

functionalities that the system needs to deliver to meet user needs.

## Non-functional requirements

in GenAl system design and may not significantly alter the initial architecture, it's crucial to identify and understand them early, as they will often shape the later stages of design, especially during performance tuning and system improvements.

Here are some questions to help you getstarted:

- Business objective: What is the primary goal of this system? What specific purpose will it serve? For example, when designing an image captioning system, it's essential to know if it will be used for generating detailed product descriptions on an e-commerce platform or for suggesting short captions for photos on social media.
- System features: What features should the system support that might influence the ML design? For instance, when designing an image generation system, it's important to know if users can provide feedback or rate the generated images, as these interactions could enhance the model. Similarly, when designing an LLM, it's crucial to know which languages should be supported.
- Data:What are the data sources?How large is the dataset? Is the data labeled?These questions are crucial because the quality and quantity of the data might influence the design.
- ·Constraints:What are the available computational resources? Will the system be cloud-based or designed to run on-device?
- ·System scale: How many users are expected to use the system? How many images need to be generated, and what is the expected growth in demand? These questions are important to clarify because a system designed to generate images for a small group of users will not require the same level of scalability as one expected to serve millions ofusers.
- Performance: How quickly should the content be generated? Is real-time generation required? Is there a higher priority on content quality or generation speed?

This list isn't comprehensive, but it provides a good starting point. Other topics, such as privacy,ethics,and data security,can also be important.

By the end of thisstep,you should be aligned with the interviewer on the system's scope and requirements. It's generally a good idea to clarify these details to ensure you're addressing theinterviewer's expectations.

## Framing the problem as an ML task

of your design.

main tool fordevelopingthesesystems.

The followingtwostepsare useful forframingyourproblem as anML task:

- ·Specify the system's inputand output
- Choose a suitable ML approach

## Specify the system's input and output

To frame the problem, you first define the system's input and output. This involves identifying the input data modality (text, image, audio, video) and the expected output. For instance, in a chatbot system, the input is the user's text query, and the output is the system's response.

Figure 1.7: Input and output of a chatbot

<!-- image -->

## Choose a suitable ML approach

After defining the input and output of the system, the next step is to choose the most suitable ML approach, This involves identifying key components of your system and selecting an algorithm that aligns with the specific needs of the problem. As illustrated in Figure is mostsuitable foryour task, MLAlgorithms There are differentways to select an appropriate ML algorithm,and the criteria for selection vary from application to application.The following steps can help you narrow down your options when choosing the most suitable algorithm:

Figure 1.8:Common ML algorithms

<!-- image -->

- 1.Discriminative vs. generative: First, determine whether the problem requires a discriminative or generative model.This can be easily determined based on the system's output. For example,in an object detection problem,where the output is the class of the input image, the task is discriminative. In contrast,designing a chatbot that produces text as output is a generative task.
- 2.Identify the task type:Next,identify the specific task type to further narrow the

Figure 1.9:Step one for choosing a suitable ML approach

<!-- image -->

image,audio,and video generation.

outputs anobject classis a classification task.

Figure 1.10: Step two for choosing a suitable ML approach

<!-- image -->

- 3.Choose a suitable algorithm:Finally,select an algorithm that is most suitable based on the requirements. Consider factors such as the ability to handle different input modalities,efficiency, and quality expectations.For example,in a text-to-image system,the algorithmmustprocesstextasinput andgenerateanimageasoutput;therefore,VAEs or GANsmight not be ideal despite their abilities to generate images.Thisstep is the ideal time to evaluate thevarious options and discuss their trade-offs.

In the upcomingchapters,wewill examinevariousML approaches used inpopularGenAl applications.

## Talking points

- ·What are the system's inputs and outputs based on the requirements?
- Which data modalities (text, image, audio, video) does the model need to understand andprocess?Howwill the model handle different modalities?
- ·Should a single model handle all input modalities, or is it more effective to use multiple models for different modalities?What are the benefits and drawbacks of using a unified model versus specialized models for each modality?
- Which generative algorithm (e.g., diffusion models, VAEs, GANs) is best suited for the task at hand, and why? What are the specific trade-offs between different algorithms in terms of quality,efficiency,and ease of use？
- rithm over another?
- modalitiesoroutputs are introduced later?

## DataPreparation

them forMLmodels.

## Datatypes

In ML, data is generally categorized into two types: structured and unstructured.

Figure 1.11:Data categories

<!-- image -->

Structured data:This type of data can be organized into tables with rows and columns. Financial records and customer data are examples of structured data.Structured data can befurther divided into the following categories:

- ·Categorical data: Data that represent distinct groups or categories (e.g., gender,color).
- ·Numerical data:Data that represent measurable quantities(e.g.,numberof items sold, houseprice).
- ·Ordinal data:Data with a predetermined order (e.g.,satisfaction ratings).

Unstructured data:Unstructured data refers to data with no underlying data schema or structure,such as text,mages,videos,audio fles,ora combination of them.For example, social media posts or emails are examples of unstructured data.

Traditional ML models are typically trained on structured data. In contrast,models that power GenAl applications primarily deal with unstructured data.As a result,thefocus of data preparation differs significantly between traditional models handling structured data and generative models working with unstructured data.Let's explore each in more detail.

## Data preparation in traditional ML

engineering and feature engineering.

Figure1.12:Data preparationprocessforstructured data

<!-- image -->

## Data Engineering

Data engineering involves building and maintaining systems for collecting, storing,retrieving, and processing data. A core component of this is ETL (Extract, Transform, Load) [21], whichrefers to the process of extracting data fromvarious sources,transforming it intoa usable format, and loading it into a data warehouse or other storage system. Data engineering ensures that data is clean,reliable,and accessible.

## Feature Engineering

Feature engineering involves selecting and extracting predictive features fromraw data and transforming them into a format usable by ML models. This process often utilizes feature stores, such as Tecton [22] or Amazon SageMaker [23], which offer a centralized platform for managing and serving features at scale.

Selecting the right features is crucial when developing and training ML models. It's important to choose features that provide the most information. The feature engineering process requires subject matter expertise and is highly task-specific. It includes techniques such as handling missing values, representing categorical features, and bucketing.

niques,refer to[24].

## Data preparation in GenAl

safe; and utilizing tools to store and retrieve the data effciently and at scale.

Figure 1.13: Data preparation process in GenAl

<!-- image -->

Let's examine the following key steps in data preparation:

- Data collection
- Data cleaning
- Dataefficiency

## Data collection

Advanced GenAl modelshave billions ofparameters that enable them to learn and generalize from data.Due to their size,these modelsrequire vast amount of training data to capture complex patterns. For instance, Llama 3 was trained on 15 trillion tokens from various internet sources-equivalent to 50 terabytes of data. To put this into perspective,a person reading nonstop at the typical rate of 250 words per minute would take around 85,oo0 years to read that amount of text. The data collection process gathers large datasets by scraping text from different sources e.g.,websites, social media,and forums).

Asmodelsgrowlarger,there'sa trend towardenhancingtrainingdatasetswithAl-generated content. This involves using existing models to create synthetic data,which is then used to train another GenAl model.

Figure1.14:AugmentingtrainingdatawithAl-generateddata

<!-- image -->

TrainingGenAl modelswithAl-generatedcontenthasseveral pros and cons.

## Pros:

- Improving data diversity: Al-generated content adds variety to existing data, thus enhancing the model's ability to generalize, especially when the original data is limited orimbalanced.
- Scalability: As demand for data grows, Al-generated content provides a scalable way tocreate large datasetsthatare difficultto gather manually.

## Cons:

- quality data can lead to the spread of biases or errors.
- Ensuring the synthetic data is diverse and representative can be challenging.
- of real-world scenarios, thereby risking the omission of important details.

thetic data,For more information,refer to[25].

## Data cleaning

or inappropriate content. We must clean the data carefully to avoid introducing biases, misinformation, or harmful material into the model, which can affect its performance. In addition,it is crucial to have data that is representative. This requires removing duplicate content and ensuring it is diverse and balanced.

Figure 1.15: Common data cleaning steps

<!-- image -->

Throughout this book, we will explore key data-cleaning techniques including filtering harmful content, detecting NsFW (Not Safe For Work),assigning quality scores,and removing duplicates.

## Data efficiency

Managing large datasets requires efficient tools and techniques for storage and retrieval. Let's look at each in detail.

## Efficient storage

Storing massive amounts of data with traditional tools can be expensive and slow. Distributed storage systems such as Hadoop Distributed File System(HDFS) [26] and Amazon S3 [27] are built to store massive amounts of data across multiple machines. These systems are particularly suited for managing large volumes of unstructured data. In addition, columnar storage formats such asParquet[28] and ORC[29] are ideal forstructured data or unstructured data that has been converted to structured form.These formats,optimized foranalytics,offerbetter compression and faster queryperformance.

## Efficientretrieval

Training an ML model requires fast data retrieval. Common techniques to retrieve data efficiently from large datasets include:

- ·Sharding: Splitting data across multiple devices allows parallel access and speeds up retrieval and processing.
- ·Indexing: Technologies such as Apache Lucene [30] or Elasticsearch [31] are used to

- 1/O delays during retrieval.

## Talking points

- are they?How large is the dataset?
- tion necessary to protect sensitive information?
- ·Bias:Are there inherent biases in the datae.g,demographic,geographicalHow d you detect and mitigate these biases to ensure fair representation?
- Data quality:How do you flter low-quality, irrelevant, or noisy data?Are there any outliersor anomalies in the dataset? How doyou handle them?
- Inappropriate data:Are there inappropriate,harmful,or NsFW contentin the dataset? What processes are in place to detect and remove such data?
- Data preprocessing:How is the data represented in a format the model can understand? If using text data,how is it tokenized and transformed into numerical format (e.g.,embeddings)? If dealing with multimodal data (e.g,images, text,audio),how do you preprocess them for the model to consume?

## Model development

Model development is a critical step in building ML systems. It involves selecting the appropriate architecture,training the model,and fnally,generating new data from th trained model. Let's dive into each of these components in more detail.

## Modelarchitecture

In this step,you should talk about the architecture of the model in detailTheremight explore diferent architectural options and weigh their advantages and disadvantages.

meet this requirement is beneficial.

Figure 1.16:U-Netarchitecture

<!-- image -->

You might be asked follow-upquestions and have to modify the architecture to support a new feature. For example, in an image generation model, you might need to modify the architecture to let users control the style of generated images.Similarly,in a textto-video model, you might be asked to control the direction of motion (e.g., left to right) during generation.These features could require adding or modifying components in the architecture to integratestylevectors ormotion information.

Let'sdiveinto areal example-theTransformer's self-attention-to show what itmeans to discuss architecture in an interview.

## Transformer'sself-attentionarchitecture

Transformers area cornerstone of modernGenAl,especiallyinnatural languageprocessing and image generation.Since their introduction in 2017 [32], they have rapidly taken over the Al community,becoming the dominant architecture fora widerange of tasks across natural language processing (NLP) [33, 12], computer vision [34], and even multimodal learning[35,36].

At the core of Transformers is the attention mechanism. Initiallyintroduced in the context of machine translation[37],the attention mechanism has become a fundamental component ofvariousneural network architectures,particularly in the Transformer model. It addresses the limitations of traditional sequence models such as RNNs and LSTMs by enabling the model to more effectively capture long-range dependencies and contextual information,

Self-attention,also known as scaled dot-product attention,is the most common form of the attention mechanism used in modern models.It enables each element in the input sequence to focuson every other element.This is done by converting the input embeddings for each token into three vectors: the query (Q), key (K), and value (V)vectors. These vectors are computed using learnable weight matrices Wo,Wk,and Wy:

<!-- formula-not-decoded -->

where Xrepresents the input sequence of embeddings.

represented as:

<!-- formula-not-decoded -->

Here,dk is the dimension of the keyvectors,and the scaling factor is used to prevent Jdk indicated by the attentionscores.

Figure 1.17: Scaled dot-product attention (Credit:[32])

<!-- image -->

## Multi-Head Attention

To capturedifferent types of relationships and contextual dependencies,the self-attention own learnable weight matrices:

MultiHead（Q,K,V)=Concat（head,head2,,head)W

where each attention head is computed independently as:

<!-- formula-not-decoded -->

different representation subspaces and capture richer dependencies.

22|Chapter 1.Introduction and Overview There is no one-size-fits-all architecture for every problem.Interviewers want to assess your understanding of different ML architectures, their strengths and weaknesses, and your ability to choose the right one based on specific requirements and constraints. In this book, we introduce various Transformer-based models through GenAl applications and explain their necessity.

Figure 1.18:Multi-head attention (Credit:[32])

<!-- image -->

## Model training

Model training is the process of adjusting the model's parameters(weights) to produce the desired output.Key aspects to discuss during model training include:

- Training methodology
- ·Training data
- ·ML objective and loss function
- ·Task-specific challenges and mitigations

Let's examine each inmore detail.

## Training methodology

Each model follows a distinct training process suited to its architecture and purpose.For example,diffusion models gradually denoise data to generate high-quality samples from noise.In contrast,GANsrely on adversarial training,wherea generator and a discriminator compete to improve over time.

Many models also undergo multi-stage training for better performance. LLMs, for in- perform well across various applications.

applications is crucial, especially when discussing them in an interview.

## Training data

used can varyacross different GenAlapplications and mayalso differinmulti-stage training approaches. For instance, when training an LLM, publicly available datasets such as Common Crawl [38] might be used during the pretraining stage, while expert-annotated, carefully curated data wouldbe used for the alignmentstage.

It's important to discuss the datasets,includinghow they are sourced,why they arevaluable for model training,and an estimate of their size for effective training.

## ML objective and loss function

ML objective is the goal of the ML task during training. For example,in an LLM,it might be to accurately predict the next token (e.g.,next-token prediction）. In contrast,in VAE, the ML objective is to reconstruct the original image.

Figure 1.19:Loss computation between training data and generated data

<!-- image -->

performance,

24| Chapter 1.Introduction and Overview functions in later chapters.

## Task-specific challenges and mitigations

Different tasks come with their own challenges that need specific solutions.For example, training large video generation models is very resource-heavy because it requires a lot of computing power and large amounts of data. This means it might not be feasible to train a video generation model without proper optimization techniques. This might include parallelization techniques [39, 40, 41], mixed precision training [42], and latent diffusion models [43]. These approaches help scale video generation models while keeping resource use and costs manageable. While we will cover task-specific challenges and mitigations in future chapters, we'll now briefly examine efficiency and optimization techniques that apply to all large-scale model training.

Three of the most common techniques for training large-scale models are:

- Gradient checkpointing
- ·Mixed precision training
- ·Distributed training

## Gradient checkpointing

Gradient checkpointing [44] is a technique to reduce memory usage during model training by saving only a selected subset of activations. During the backward pass, the missing activations arerecomputed.This reduces memory usage significantly,which is particularly useful for training large models with limited GPU memory.

## Mixed precision training

Mixed precision training is a technique that uses both 16-bit (half-precision) and 32-bit (single-precision) floating point numbers to speed up model training and reduce memory usage. It maintains the accuracy of training while improving efficiency by performing most calculations at lower precision; crucial operations are performed at higher precision when needed.

Automatic mixed precision (AMP) [45] is a specific implementation of mixed precision training provided by frameworks such as PyTorch and TensorFlow.AMP automatically handles the transition between half and single precision,optimizing where to use each precision type and applying scaling techniques to maintain numerical stability during training.

## Distributed training

As models grow in size and complexity, training on a single machine becomes infeasible. Distributed training techniques enable efficient training of large models by utilizing multi- ple machines or devices in parallel.

Figure 1.20:Parallelism techniques for distributed training

<!-- image -->

## Common parallelism techniques are:

- Data parallelism
- ·Model parallelism
- ·Hybrid parallelism

## Data parallelism

is large,as processing the data in parallel is efficient and speeds up training.

Figure 1.21:Data parallelism

<!-- image -->

There aretwoprimarymethodsforupdatingmodel parameters across the devices:

- ·Synchronous: In this approach, all devices complete their computations and send gradients to theparameterserver.Theparameterserverwaitsuntil it hasreceived gradients from every device, and then it aggregates and updates the model before sending the updated parameters back to all devices. This ensures consistency, as the devices alwaysworkwith the sameversion of the model.However,it canbe slower because updatinghas towaitfor the slowest device.
- Asynchronous:With asynchronous updating,each device sends its gradients to the parameter server as soon as it finishes processing its portion of data,and the parameter server updates themodel immediately uponreceivinggradientsfrom any device and sends the new parameters to all devices. This approach can be faster since the devices are working independently,but it can lead to inconsistency as devices might beworkingwith slightly differentversions of the model at any given time.

To learn more about data parallelism, refer to [39].

## Model parallelism

when the model is too large to fit into the memory of a single device.

Model parallelism can be further divided into two types:

- Pipeline parallelism (inter-layer)
- ·Tensor parallelism(intra-layer)

it sends the input tensor's gradient back to the preceding device.

Figure 1.22:Splitting model layers across devices

<!-- image -->

aboutPP,refer to[40,41], multiplication operation, different parts of the matrix can be processed in parallel across more information, refer to [46].

Figure 1.23:TP splitting tensor to chunks

<!-- image -->

memory usage by distributing the computational load across multiple devices. lf you are interested to learn more about TP and its variants (e.g., sequence parallelism),refer to [47,48].

## Hybrid parallelism

Hybrid parallelism combines data and model parallelism to train large models more effciently acrossmultiple devices.This approach reduces memory usage by distributing both the model and data across devices. It also enables scaling to a larger number of devices, making itpossible to train very large models that traditional parallelism methods cannot handle.

In addition to hybrid parallelism,techniques like ZeRO (Zero Redundancy Optimizer)[49] from Microsoft and FSDP (Fully Sharded Data Parallel)[5o] from Meta further optimize resource utilization and communication efficiency. These methods reduce redundancy models.

training,refer to the provided references.

## Model sampling

search [51], and top-k sampling [52] each have their strengths and weaknesses. Beam search,orinstanceendsroducecoherent and relevanttxt,utimaitdivrsit

Figure 1.24:Sampling new data from a trained model

<!-- image -->

cons,and to choose the one that suits the system you are designing.

## Talking points

- architecture and why?
- process work(e.g,diffusion process,adversarial training)?

- and cons of each objective, and how do they impact model performance?
- Do you use a single loss function or multiple ones? If multiple, how do you combine them to optimize the training process? What is the purpose of each loss function?
- Training challenges and mitigations:What are typical training challenges specific to the chosen ML algorithm? How can these challenges be mitigated to ensure effective
- Training efficiency: What are the main techniques to improve training efficiency? How does distributed trainingwork,and what benefits does it bring?How does AMP enhance training speed and efficiency? How does data, tensor, and pipeline parallelism work?
- ·Sampling: How do different sampling methods (e.g., top-k, top-p) work? What are the pros and cons of each? How do they affect the quality and creativity of the model's output? What methods can you use to make the sampling process faster without compromising quality?

## Evaluation

After developing a model, the next critical step is evaluation. This involves using various metrics to assess the performance of the ML model. In this section,we will explore two evaluation methods: offline and online evaluation.

## Offline Evaluation

Offline evaluation is the process of assessing the performance of a model or system using pre-collected data without deploying it in a real-time environment. This approach is critical to ensure the model is effective before it is used by the users.

Offline evaluation differs between discriminative and generative models. The goal of discriminative models is to make predictions on an evaluation set; the evaluation compares those predictions to the ground truth.Traditional metrics such as accuracy,precision, and recall are used to quantify how well the model performs based on these comparisons. Table 1.2 lists common metrics for different discriminative tasks,which are thoroughly explored in [1].

Table 1.2:Popular metrics in discriminative tasks

| Task           | Metrics                           |
|----------------|-----------------------------------|
| Classification | Precisionalorecuracnfusina        |
| Regression     | MSE,MAE,RMSE                      |
| Ranking        | Precision@k,Recall@k,MRR,mAP,nDCG |

for different generative tasks.

Table 1.3:Popular metrics in generative tasks

| Task             | Metrics                           |
|------------------|-----------------------------------|
| Text Generation  | Perplexity,BLEU,METEOR,ROUGE,IDEr |
| Image Generation | FID,IS,KID,WD,PP,PIS              |
| Text-to-Video    | FVD,CIPSr,FID,PIPS,ID             |

In interviews,it's essential to assess the generated content from multiple angles. For example,in a text-to-image generation model,it'simportant to ensure the generated image is both high-quality and that it aligns with the given text prompt. Similarly, in a chatbot, the model's capability should be measured across different tasks such as mathematics, common-sense reasoning,and code generation. Throughout this book,we take a deep listed in Table 1.3.

## Online Evaluation

impact.

Table 1.4:Common metrics for online evaluation

| Metric                   | Description                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------|
| Click-Through Rate(CTR)  | Percentage of users who click on content or suggestions.                                                         |
| Conversion Rate          | Percentage of users who complete a desired action (e.g, purchase,subscription) after interactingwith the system. |
| Latency (Inference time) | Time takenby themodel togeneratecontent.                                                                         |
| Engagement Rate          | Measure of userinteraction,such astime spentengaging withthesystem.                                              |
| RevenuePerUser           | Averagerevenuegeneratedper user.                                                                                 |
| Churn Rate               | Percentage of userswho stopusing the system overagiven period.                                                   |
| User Satisfaction        | Directfeedback from users on their experiencewith Al-generated content.                                          |
| User Retention           | Percentageof userswho continue to use the system overa specific period.                                          |
| Completion Rate          | Percentage of tasks(e.g.,text completions,image generations) successfully completed by themodel.                 |

## Talking points

Here are some key talking points for the evaluation stage:

- Offline metrics:Which offline metricsbest evaluate the quality and accuracy of the generative model? How do these metrics measure the diversity, realism, and coherence of generated outputs?
- Online metrics:Which metrics are crucial for assessing the effectiveness of the generative model in a live production environment? How do these metrics align with the business goals,such as enhancing user creativity,boosting engagement,or driving product innovation?
- Bias: Generative models may unintentionally reflect societal biases in relation to sensitive attributes such as gender or race. How can you evaluate the model's bias?
- Robustness and security:How resilient is the generative model to adversarial attacks such as intentionally misleading inputs designed to exploit model weaknesses?
- Human evaluation:For generative models,especially in creative fields (e.g.,text generation,image synthesis)humanfeedbackisvital.How canhumanreviewerscomple mentautomaticevaluation?Whatmethods(e.g.,surveys,A/Btesting,expertreviews) will best assess the model'sperformance? How can you mitigate the effects of subjectivity among different reviewers?

## Overall ML system design

## Talkingpoints

- post-processing,and any necessary upscaling or quality enhancement models?
- system?For instance,how does the system ensure that generated content is safeand appropriate for users?
- User feedback and continuous learning: How does the system incorporate user feed back to continuously improve model performance? Discuss the feedback loop mechanisms that allow for finetuning the model. What systems are in place for retraining models with updated data to improve accuracy and relevance over time？
- Scalability: How does the system scale as demand increases? What cloud or hard wareresources are utilized,and how is resource allocation managed efficiently?How do components such as load balancers,distributed inference,and model parallelism contribute to system scalability?
- Security considerations:How does the system ensure user privacy,especially when dealing with sensitive data or generating personalized content?What security protocols are mplemented to protect against adversarial attacksmodel tamperingor data leakage?
- duce harmful,biased,or inappropriate content?
- creating deepfakes,misinformation,or inappropriate content?

## Deployment andmonitoring

The final step is to deploy the system to production and serve millions of users. Once the system is deployed, it can fail for many reasons. Monitoring refers to the task of tracking, measuring, and logging different metrics to detect system failures when they occur, so they can be fixed as quickly as possible. However, since this topic is broad and not specific to GenAl or any particular task, we won't go into detail in this book. We encourage readers to refer to [53] or the "ML System Design Interview" book [1] for a deeper exploration.

## Summary

In this chapter,we provided an overview of GenAl and introduced a framework for approaching a GenAl system design interview. While some details are specific to GenAl, many concepts apply broadly across Al system design. We focus on aspects unique to GenAl, avoiding general topics common to all Al systems such as deployment, infrastructure, and monitoring.

Finally, not every engineer is expected to be an expert in all areas of the GenAl life cycle. Different roles and companies may emphasize various aspects, such as infrastructure, monitoring, or LLM development. This framework helps candidates to understand the expectations and adjust their answers accordingly based on the interview's focus.

Now that you understand these fundamentals, we're ready to tackle some of the most common GenAl system design interview questions.

## Gmail Smart Compose

Gmail's Smart Compose feature [1] assists users by suggesting the next few words as they write an email. This chapter explores this feature and examines the Transformer architecture thatpowersmostgenerative systems.

Figure 2.1: Gmail's Smart Compose feature

<!-- image -->

## Clarifying Requirements

Here is a typical interactionbetween a candidate and an interviewer:

Candidate: Different users might have different writing styles. ls the system expected to makepersonalized suggestions?

Interviewer:For simplicity,let'snot include personalization.

Candidate: Should the system suggest the next few words only when it is confident in its prediction?

Interviewer: Yes.

Candidate:The email dataset must be suffhiciently large to train a model. Do we know the approximatesizeofthedata?

Interviewer: Assume our dataset consists of around one billion email messages.

utilize the email's body as the context?

can expand the context to include other relevant information.

Candidate: What languages should the system support?

Interviewer: Let's begin with English.

Candidate:Do we need to ensure the system is not biased?

make biased assumptions in providing its suggestions.

this feature?

Interviewer: Gmail has about 1.8 billion users, and a single user can send as many as 500 emails in a day.We do care about the computing costs,but let's focus on developing the system first.We can optimize for efficiency in future iterations.

Candidate: Should the system make real-time suggestions?

Interviewer: Yes. The expected latency should be imperceptible; something around 100 milliseconds should be fine.

## Frame the Problem as an ML Task

learning the task.

## Specifying the system's input and output

next.

Figure 2.2:Input and output of the Smart Compose system

<!-- image -->

## Choosing a suitable ML approach

Transformers[3].

Transformers provide several advantages over RNNs, with two main benefits being:

- ·Parallelism:In an RNN,computations from one time step are carried forward and used in the next,creatinga time-dependent chain of operations.Transformers,on the other hand,can process all input tokens simultaneously through their self-attention mechanism.
- ·Better handling of long sequences: Transformers use self-attention mechanisms to focus on any part of a sequence,regardless of distance.In contrast,RNNs,struggle with long-range dependencies because of their sequential structure and thevanishing gradient problem.

Due to these advantages,Transformershave shown outstandingperformance in text generation tasks and are, thus,used in most generative systems nowadays. Therefore,we choose Transformers to build the Smart Compose feature.

Table 2.1:Comparison of RNN and Transformer architectures

| Feature             | RNN (GRU[4],LSTM[5])                                                         | Transformer                                              |
|---------------------|------------------------------------------------------------------------------|----------------------------------------------------------|
| Architecture        | Simple                                                                       | Complex                                                  |
| Training efficiency | Inefficientdue to sequential Efficient due to parallel processing processing |                                                          |
| Effectiveness       | Low as it struggles with long sequences                                      | High as it handles long sequences                        |
| Scalability         | Limited scalability                                                          | Highly scalable                                          |
| Applications        | Simple taskssuch as timeseries modeling                                      | Complex tasks such as language completion or translation |

While Transformers are more parallelizable due to their lack of strict sequential dependencies,their self-attention mechanism has a computational complexity of O（n²),where n is the sequence length. This complexity arises because the self-attention mechanism requires the calculation of attention scores between every pair of tokens in the sequence. Various techniques are introduced to reduce the complexity of attention. To learn more, refer to Group Attention [6] and Flash Attention[7].

## Data Preparation

ML model.First,let's briefly review the available data.

diverse vocabulary,syntax,and contexts.

Those hours that with gentle work did frame The lovely gaze where every eye doth dwell Will play the tyrants to the very same, And that unfair which fairly doth excel: For never-resting time leads summer on To hideous winter and confounds him there, Sap checked with frost and lusty leaves quite gone, Beauty o'er-snowed and bareness every where: Then were not summer's distillation left A liquid prisoner pent in walls of glass, Beauty's effect with beauty were bereft, Nor it nor no remembrance what it was.

But flowers distilled though they with winter meet,

Leese but their show, their substance still lives sweet.

Then let not winter's ragged hand deface, In thee thy summer ere thou be distilled: Make sweet some vial; treasure thou some place, With beauty's treasure ere it be self-killed: That use is not forbidden usury, Which happies those that pay the willing loan; That's for thy self to breed another thee, Or ten times happier be it ten for one, Ten times thy self were happier than thou art, Music to hear, why hear'st thou music sadly? Sweets with sweets war not, Joy delights in joy： Or else receiv'st with pleasure thine annoy? If the true concord of well-tuned sounds, By unions married do offend thine ear, They do but sweetly chide thee, who confounds In singleness the parts that thou shouldst bear: Mark how one string sweet husband to another, Strikes each in each by mutual ordering; Resembling sire, and child, and happy mother, Who all in one, one pleasing note do sing:

whose speechless song being many, seeming one, Sings this to thee,'Thou single wilt prove none

9

Is it for fear to wet a widow's eye, That thou consum'st thy self in single life? Ah,if thou issueless shalt hap to die, The world will wail thee like a makeless wife, The world will be thy widow and still weep, That thou no form of thee hast left behind, When every private widow well may keep, By children's eyes, her husband's shape in mind: Look what an unthrift in the world doth spend Shifts but his place, for still the world enjoys

Figure 2.3:Example of general data from Shakespeare

is stored for each email message.

Table2.2:Exampleofemail data

|   Email ID | Sender          | Recipient          | Subject          | Body                                              |
|------------|-----------------|--------------------|------------------|---------------------------------------------------|
|       4953 | john@gmail.com  | mike@yahoo.com     | Catchup?         | Hey Mike,let's catch up this Sat.....             |
|       9356 | kkart@gmail.com | cs382@stanford.edu | Project Deadline | Hi TA,Ihope you are well.Iam writing to you to... |

Raw text inbothgeneral data and email data isoften noisy and inconsistent,whichcan degrade the model performance.Additionally,MLmodelsrequire data to beinanumerical format.Forthesereasons,rawtexthas tobeprepared using thefollowing twokey steps:

- Textcleaning andnormalization
- ·Text tokenization and token indexing

## Text cleaning and normalization

## Text cleaning

Text cleaningremoves unnecessary orirrelevant information.Common methods include:

- Removenon-English text:Uselanguage identificationmethods such as[8,9] to identify and remove non-English text from general and email data.
- Remove confidential information:Emails may contain confidential information such as phone and credit card numbers. These details must be removed to prevent the model fromlearningorexposingthem later.Wereplacepersonal names,URLs,email addresses,and phone numbers with placeholder characters.For example,replace "john@gmail.com"with"###@gmail.com."
- Remove irrelevant characters or symbols:Remove unnecessary or irrelevant characters and symbols thatdonotcontributeto the meaning.Forexample,symbolssuch as """m, or emojis are removed, as they do not typically change the meaning of text.
- Remove duplicateddata:Duplicate datarefers toidentical textfrom differentsources thatappear multiple times in the dataset.We remove duplicates toprevent the model frombecomingbiased andskewing the model'slearningprocess.

## Textnormalization

Textnormalization transforms textintoa consistentformat.For example,it converts different ways of writing a phone number-such as"(123)456-7890,"123.456.7890.and "123-456-7890"-into a standard format,forexample,"1234567890."Textnormalization ensures consistency and reduces complexity in text data.

Next,we convert theraw text into a sequence ofnumbers through text tokenization and token indexing.

## Text tokenization and token indexing

former model expects:a sequence of numbers.

Figure 2.4:Converting raw text to a sequence of numbers

<!-- image -->

Let's examine each step in more detail.

## Text tokenization

Text tokenization is the process of splitting text into smaller units called tokens.Figure 2.5 shows how OpenAl's GPT-4 tokenizes the sentence "Let's go to NYC". 1

Figure 2.5:Example of GPT-4 tokenization

<!-- image -->

algorithms are divided into three categories:

- Character-level tokenization
- Word-level tokenization
- Subword-level tokenization

Visit https://platform.openai.com/tokenizer to see examples of different tokenizers.

44|Chapter 2.Gmail Smart Compose

interviews.Let's delve into them.

## Character-level tokenization

Character-level tokenization breaks text down into a set of characters. It is simple to implement, but diffcult for the model to learn meaningful representations for each token. For example, it's harder to learn a meaningful representation for the letter "g" than for the word "go, because "go"has a clear meaning, whereas "g" does not. Because of this, character-level tokenization often results in a loss of performance.

Figure 2.6:Example of character-level tokenization

<!-- image -->

## Word-level tokenization

Word-level tokenization breaks text into individual words. While there are different algorithms forword-level tokenization,a simple algorithm is to split the text using itswhitespaces.

Figure 2.7:Example of word-level tokenization

<!-- image -->

The advantage of word-level tokenization is that it is simpler for the model to learn meaningful representations for each token.However,the main disadvantage of word-level tokenization is that it typically leads to avery largevocabularysize.Forexample,Transformer- more costly to train than character-level tokenization.

character-level tokenization.

## Subword-level tokenization

and "ly" are more frequently used in text data,making it easier for the model to leana meaningful representation for each.

Figure 2.8:Example of subword-level tokenization

<!-- image -->

to represent unfamiliar words by decomposing them into known subwords.

Table 2.3:Comparison between different tokenization categories

| Characteristics       | Character-level                       | Word-level                       | Subword-level                           |
|-----------------------|---------------------------------------|----------------------------------|-----------------------------------------|
| Granularity           | Individualcharacters                  | Individualwords                  | Subwords                                |
| Vocabulary size       | Small                                 | Large                            | Moderate                                |
| Algorithm complexity  | Simple                                | Simple                           | Complex                                 |
| Handling unseen words | Decomposes unseen wordsintocharacters | Cannot easily handle unseenwords | Decomposesunseenwords intoknownsubwords |
| Vocabulary size       | 100-1,000                             | 300,000+                         | 50,000-150,000                          |
| Performance           | Poorperformance                       | Highperformance but notpractical | High performance and practical          |

## Which tokenization is suitable for the Smart Compose feature?

Most state-of-the-art language models use subword-level tokenization algorithms such as Byte-Pair Encoding(BPE)[11]and SentencePiece[12]. Thesealgorithms are more efficient and can effectively handle multiple languages. For example, OpenAl's GPT-4 uses a variant of BPE[13],and Google's Gemini usesSentencePiece[14].

Given the effectiveness ofsubword-level tokenization,we use it as the text tokenizer for the Smart Compose feature. We rely on popular Python libraries such as Tiktoken[13] by OpenAl orSentencePiece[15] by Google toperform text tokenization.These librariesare implemented reliably and they supportvarious tokenization algorithms.

In Chapter 3, we will dive into BPE and explore its algorithms. To learn more about subword-level tokenization algorithms,refer to[16].

## Token indexing

Token indexing is the process of converting textual tokens into integer numbers.

Toprepare for token indexing, the tokenization algorithm first builds a vocabulary-a collection of allunique tokens-from the training text data and then stores it in a table. Figure 2.9 shows examples of vocabularies for different tokenization categories. The order and ID values are chosen arbitrarily for demonstration purposes.

| Token   |   ID |
|---------|------|
| a       |    0 |
| b       |    1 |
| A       |   26 |
| B       |   27 |
|         |   57 |
| <SPACE> |  105 |

Character-level Vocabulary Word-level Vocabulary Subword-level Vocabulary

| Token   |     ID |
|---------|--------|
| a       |      0 |
| about   |      1 |
| after   |      2 |
| all     |      3 |
| also    |      4 |
| zebra   | 270030 |
|         | 270131 |

| Token   |    ID |
|---------|-------|
| the     |     0 |
| of      |     1 |
| home    |     2 |
| ##ing   | 50252 |
| ##ed    | 50253 |
| ##able  | 50254 |
| <EOS>   | 50255 |
| <SPACE> | 50256 |

Figure2.9:Examplesof differentvocabularies

Once the tokenization algorithm has built the vocabulary,we can convert any token into a number and any number back into a token.Figure 2.10 shows token indexing using the GPT-4vocabulary[17].

<!-- image -->

GPT-4 Vocabulary

| Token   |    ID |
|---------|-------|
| i       |     0 |
| 's      |   596 |
| go      |   733 |
| Let     | 10267 |

Figure 2.10:Example of token indexing

To summarize the data preparation step, we first clean and normalize the text data to tokenization algorithm such as BPE to tokenize the text into textual tokens (subwords) and then replace each token with its numerical index. These steps ensure our training data is now represented in a numerical format that the ML model can use.

## Model Development

The Smart Compose feature isa textgeneration taskin whicha Transformermodel predicts how email sentences are likely to be completed. In this section, we explore the details of the Transformer architecture, training strategies, and sampling methods to develop the textgenerationmodel.

## Architecture

The Transformer architecture, introduced in the paper "Attention Is All You Need" [3], is designed to process sequences. This makes it ideal for tasks that require understanding a text and therelationships between itswords.Forexample,in the Smart Compose feature, the model processes the sequence ofwords already entered by the userso itcan suggest the nextwords.

Transformershave threeprimary variations:

- Encoder-only
- Decoder-only
- Encoder-decoder

Each variation has minor architectural differences that make them suitable for specific tasks. Let's briefly explore each variation and its applications.

## Encoder-only

An encoder-only Transformer isused for tasks that require understanding the overall meaning of a text. It processes the input sequence as a whole and makespredictions about it. For instance,in a sentiment analysis task,an encoder-only Transformer predicts the sentiment of the input sentence.

Figure 2.11: Encoder-only Transformer for sentiment analysis

<!-- image -->

erating new content. Google's BERT [18] is a wel-known example of an encoder-only Transformer. However, these models are not typically used to generate new sequences. Decoder-only Transformers, on the other, are specifically designed for that purpose.

## Decoder-only

A decoder-only Transformer processes the input sequence and generates a new sequence iteratively.

Figure 2.12:Decoder-only Transformer for text completion

<!-- image -->

## Encoder-decoder

processed information to generate the output sequence An encoder-decoder Transformer is particularly suited for tasks where the output is a transformation of the input. For example, in a language translation task, the input sentence in one language is transformed into an equivalent sentence in another language. We'll examine this architecture in Chapter3.

Figure 2.13:Encoder-decoder Transformer used for language translation

<!-- image -->

Figure 2.14 below shows commonly used models that employ differentvariations of Transformers.

Figure 2.14:Popular models for each variation of Transformer

<!-- image -->

## feature?

a given sequence.

tectures,refer to[21] and[22].

A decoder-only Transformer consists of the following components:

- Text embedding
- Positional encoding
- Transformer
- ·Prediction head

## Text embedding

The text embedding component converts each token ID into a fixed-length vector called an"embedding Embeddings are typically stored in a table, as shown in Figure 2.15,and learned during the trainingprocess.

<!-- image -->

Embedding table (dimension=3)

Figure 2.15:Embedding table representing tokens

During data preparation, we tokenized the text and converted tokens to IDs. However,

- Sparsity: The vocabulary typically includes tens of thousands of token IDs. Representing these IDs using one-hot encoding results in sparse, high-dimensional data, which isinefficient.
- Lack of semantic information: Token IDs are arbitrary and do not capture any relationships between words. For example, the words "happy" and "joyful" might be close in meaning, but their token IDs may not reflect this similarity.

The text embedding component addresses both of these limitations by converting token IDs into learned embeddings. Since the embeddings are dense vectors in a lowerdimensional space, sparsity is no longer a concern.

In addition,since the embeddings are learned during model training,they capture semantic meanings. For example, the embeddings for "happy" and "joyful" will be closer together in the embedding space than those for "happy" and "sad," as shown in Figure 2.16.

Figure 2.16:Word embedding similarities (visualized in 2D for simplicity)

<!-- image -->

## Positional encoding

mula for attention,

<!-- formula-not-decoded -->

impacts themodel's ability to understand or generate coherent text.

To overcome this limitation, positional encoding provides the Transformer with position information for each token in the input sequence.Without positional encoding,the model treats the input sequence as a bag of words,which is problematic.With positional encod ing,each token's position is encoded using a positional encoding function,

<!-- formula-not-decoded -->

where f() is the positional encoding function and i is the position of the token. This allows the model to distinguish between"use the variable, then initialize it" and "initialize thevariable,then use it."

Figure 2.17: Adding positional information to the Transformer's input sequence

<!-- image -->

Positional encodingcan be achieved through two common methods:

- Fixed positional encoding
- Learned positional encoding

## Fixed positional encoding

This method uses a fixed function to map a position (an integer) to a fixed-size vector. The original Transformer paper introduced the sine-cosine function at different frequencies as itspositional encodingfunction.

Figure 2.18:Sine-cosine positional encoding formula

<!-- image -->

dings so they can be added together (see Figure 2.17).

Figure 2.19:Example of sine-cosine positional encoding

<!-- image -->

Let's take a look at the pros and cons of fixed positional encoding.

## Pros:

- Efficiency: Fixed encodings do not add extra trainable parameters to the model. This makes them computationally more efficient.
- training data.

## Cons:

- sition, thus limiting their applicability to sequences below that maximum.
- performance.

## Learned positional encoding

able parameter, and it is optimized alongside the model's other parameters.

Figure2.2O:Trainablematrixrepresentingpositional encodings

<!-- image -->

Learned positional encoding has the following pros and cons.

## Pros:

- Optimal performance: Since the embeddings are learned based on the training data, learned positional encoding can lead to optimal position representation for the specific task.

## Cons:

- Inefficiency:Requires additional parameters to be learned during the training,which can increase the training time and computational cost.
- ·Lack of generalization:Learned embeddings may overfit to specific sequence lengths seen during training. If the model mainly sees sequences of a certain length during training,it may not effectively represent other positions.This affects the model's ability togeneralize across diversepositions.

In summary,the choice between learned and fixed positional encodings depends on the constraints of the task, including the expected variability in sequence lengths. Some papers,including the original Transformer paper, employ fixed positional encoding due to its efficiency and better generalization.Following that,we employ fixedpositional encoding, such as sine-cosine encoding,to train the Smart Compose feature.

## Transformer

The Transformer component takes a sequence of embeddings as input and transforms them into an updated sequence of embeddings.

lowing:

- use throughout the rest of this book.
- between,to each embedding in the sequence independently.

"Attention Is All You Need"[3] and [21],

Figure 2.21:A simplified Transformer structure

<!-- image -->

## Prediction head

The prediction head-the final component in a decoder-only Transformer-translates the Transformer's output into probabilities for every token in the vocabulary (Figure 2.22). These probabilities are used to choose the most likely next token.

Figure 2.22: Prediction head output probabilities

<!-- image -->

## Training

Training adjusts the decoder-only Transformer's parameters using email data. Once the training process is complete,the model can suggest likely completions.

However,directly training the model on a task-specifc dataset, such as email data, isnot a good strategy.This direct training has several challenges:

- ·Lack of large training data: Task-specific datasets are usually limited in size. This limitation can hinder the model'sability to learn effectively.
- Risk of overfitting:When a model is trained on a task-specific dataset,it runs a high risk of overfitting. Overfitting occurs when a model memorizes the training data to the extent that it cannot generalize to unseen data.
- Expensive and lengthy training: Training a large model from scratch requires signifcant computational resources and time.This is because the model has to learn different aspects of language,which is a complex and resource-intensive process.

completion).

Instead,it adjusts itspretrainedweights,which ismore efficient.

Figure 2.23:Two-stage training strategy

<!-- image -->

tives,and loss functions for each.

## 1.Pretraining

syntax,common knowledge,and language structures.

## Pretraining data

since 2008.

## ML objective and loss function

case of text generation, the most commonly used ML objective is next-token prediction." In this ML objective, the model is tasked with predicting the next token given a sequence of previous tokens.Forexample,in the sentence"hope you are, themodel should predict a high probability for "well" as the next token.

Figure 2.24: Probability distribution in next-token prediction

<!-- image -->

Next-token prediction is well suited for text generation tasks because, after the training process, the model can construct sentences incrementally.For example,given the input "1 ordered food because I,"the model might predict"was"as the next word. Subsequently, this process repeats with the new sequence "I ordered food because I was, leading to the next prediction, perhaps,"hungry." This iterative process continues until the model predicts"(Eos),"a special token that indicates the end of the sequence.Figure 2.25 shows the incremental process of generating text using next-token prediction.

Figure 2.25:Incremental generation of text

<!-- image -->

To optimize the model for correctlypredicting the next token,we define a loss function to guide the training process. Cross-entropy loss [24] is a commonly used loss function for the next-token prediction objective.This loss function measures the differences between In practice,the model processes all token lengths within a sequence in parallel. This allows it to compute the loss for each token position simultaneously.Parallelizing this step speeds up training by handling multiple tokens at once,instead of sequentially.

Figure2.26:Losscalculation

<!-- image -->

Figure 2.27:Parallelizing loss computations over different lengths

<!-- image -->

## 2.Finetuning

its language understanding from the pretraining stage but adapts to the nuances of the task.

## Finetuning data

We use a dataset of approximately one billion email conversations,as specified in the tones, and specific vocabularies that are more common in email conversations.

## ML objective and loss function

In the finetuning stage,both the ML objective and loss function remain unchanged.The ML objective is next-token prediction,and the cross-entropy loss function guides the training process. The only difference from the pretraining stage is that the loss is calculated based on email data,focusing on predicting the next token in an email context.

Figure 2.28:Example of email completion

<!-- image -->

However,relying on the email's body as the sole input is not very effective,because t is not always possible to predict the next token this way. Imagine a user who wants to reply to an email from John.When the user types"Dear" the model should, ideally, suggest "John However,if that information is not provided as input, the model cannot predict "John"as the likely next token.

To address this limitation,we include more information in the input.For example,we can use the email's subject,therecipient,and previous emails,if available.This adds depth o the context and helps the model make more relevant predictions.

<!-- image -->

## Combining various inputs

types like text,images,or tables.

handle diverse inputs with a unified architecture, thus streamlining development and enhancing the versatility of GenAl systems.This decoupling is done through techniques like prompt engineering[25]. In Chapter 6,we examine prompt engineering in detail.

Figure 2.30:Combining various inputs in traditional ML vs.GenAl era

<!-- image -->

input structures and still produces reliable results.

Output:collaboration

<!-- image -->

Output: later this week?

Figure2.31:Examples of combining textinputs

## The benefits of two-stage training

The two-stage training strategy has several benefits, including:

- ·Adaptability: The same base model obtained from the pretraining stage can be adapted for differenttasks.
- Improved generalization:Pretraining on large and diverse text data enables themodel to develop a broad understanding of language. This helps to generalize better to various tasks.
- Fast finetuning:The model learns general knowledge during the pretraining stage. Thismakes the subsequent finetuning process faster.
- Handling data scarcity:For tasks where large datasets are unavailable,the knowledge gained duringpretrainingcan compensate for this lackofdata.This allows themodel toperform well even with limited task-specific data.
- ·Mitigating overfitting: If we train a model from scratch on a smaller, task-specific dataset,thereis a risk itwill overfit.In two-stage training,pretraining actsasregularization.The model first learns to understand language broadly before focusing on the specifics of a particular task.
- ·Resource optimization:By separating the training process into two stages,we perform the computationally expensive pretraining once and can reuse the same model to adapt to different tasks.Thisreduces computational costs since we do not need to repeat the pretraining stage for each task.

## Sampling

Generative models are trained to capture the underlying distribution of the training data. Once trained, these models can generate new samples that are similar to the data they were trained on. Sampling is the process of using a trained generative model to generate newdata.

In the context of Smart Compose, sampling involves generating a likely email completion model predicts the(Eos)token.

Figure 2.32:Sampling email completion token by token

<!-- image -->

There are primarily two types of strategies to generate new text in generative models: deterministic and stochastic. Let's have a look at each.

## Deterministic

Deterministic methods generate text in a deterministicway,that is,without randomness or variability in the output.For example,at each step of token generation, the model se lects the token with the highest probability from the predicted distribution. This method ensures that the generated text will always be the same for a given input, thus providing consistency and reproducibility. Figure 2.33 ilustrates greedy search," a simple deterministic method to generate text by iteratively choosing the next token based on the highest predictedprobability.

## Pros:

- Consistency: The generated text is always the same for the same input -a desirable property for systems requiring predictable results.
- ·Predictable outputs: Fewer surprising outputs are generated because it always chooses the most probable token at each iteration.

## Cons:

- ·Lack of diversity: The model may miss less probable but more interesting tokens; thus,there will be less creativity in the generated text.For example,when generating a story,the model always choose the most common phrases,resulting in a predictable but less interesting narrative.
- Repetitive text:The text may become repetitive,as the same high-probability token is always selected. For example, if the model generates a lengthy article, it might repeatedly use certain phrases. A real example of this is shown in Figure 2.34.

Figure 2.33:Greedy search

<!-- image -->

## Pros;

- ticularly useful in applications such as dialogue generation.

Figure 2.34:Text generated by GPT-2 language model using greedy search

<!-- image -->

## Stochastic sampling

Stochastic sampling methods introduce randomness into the generation process. For example,at each step of token generation,the model samples from the predicted distribution based on the probabilities assigned to each token. This means that each time text is generated,even with the same initial inputs,the generated text may vary.

Figure 2.35 shows two instances of sampling using the same initial token"How." The first time,the sequence of generated tokens leads to"How are you;the second time,a different sequence is generated using the same initial token due to the randomness inherent in sampling.

Figure 2.35:Stochastic samplingrandomness

<!-- image -->

## Cons:

- Inconsistency:The output mayvary each time a text is generated.Thisisless suitable for applications that require precise,repeatable results.
- Unexpected outputs: The randomness can lead to unexpected variations in the generated text,which might be inappropriate.

## Whichgeneration method issuitableforthe Smart Compose feature?

For Smart Compose, deterministic methods are preferred for several reasons:

- Consistency: Consistency in generated text is crucial for applications such as email completion,for which users expect predictable and reliable suggestions. Utilizing a deterministic method means that users won't see dramatically different suggestions each time they begin to type the same thing.
- Better handling of common phrases: Deterministic methods are typically preferred in an email context, as more likely completions are prioritized over the novelty that stochasticmethods offer.
- Reduced risk of inappropriate suggestions: Stochastic methods might occasionally generate inappropriate suggestions due to their inherent randomness. This behavior is not desired in an email completion feature.

These reasons highlight why deterministic methods are preferred in applications requiring consistency such as email completion. Now that we have chosen deterministic text generation, let's examine two primary algorithms:

- ·Greedy search
- ·Beam search

## Greedy search

Greedy search is the simplest deterministic algorithm.It always selects the token with the highest probability as the next token.Aswas shown in Figure 2.34,greedy search can lead to repetitive patterns in the generated text. This occurs because it follows a narrow path based on the highest probability tokens without considering alternative paths that might lead to more coherent sentences. Due to this limitation, greedy search is rarely used in practice.

## Beam search

Beam search [26] is a popular deterministic algorithm for generating text from a trained model. The core idea is to track multiple potential sequences of tokens simultaneously. At each step,the model calculates the probabilities for the next possible tokens for each sequence and selects the "top-k" most probable sequences. The value of k, known as beam width,is configurable.

Figure 2.36: Beam search calculating the top three most probable sequences (beam width=3)

<!-- image -->

Here is a brief step-by-step process for generating text using beam search assuming a beamwidthof3:

- the top three tokenswith thehighestprobabilities.
- bilitiesof thenexttoken.

Figure 2.37:First iteration of a beam search with beam width=3

<!-- image -->

The expansion and pruning steps are repeated until all three potential sentences reach the(Eos) token or a maximum length. Once the beam search algorithm has stopped,we select the sequence with the highest cumulative probability as the output.

Beam search is effective in practice since it tracks several potential sequences simultaneously instead of the most probable sequence. However, beam search has two main drawbacks:

- Limited diversity: Beam search often leads to similar outputs, which is not ideal for applicationsrequiring diverse responses.
- ·Struggle with long sequences: Beam search struggles with longer sequences because tracking too many sequences simultaneously can become computationally expensive.

The suggestions made by the Smart Compose feature are typically short;hence, capturing long-range dependencies is less critical. In addition, diversity in the email completions is not desired.For these reasons,we choose beam search as the primary sampling algorithm for generating suggestions,

## Evaluation

Evaluation is essential in ML system design interviews. Interviewers will check if candidates can effectively test and validate the ML system they design. An ideal answer should cover both online and offline evaluations and discuss popular metrics for measuring a model's performance in each setting.

## Offline evaluation metrics

data.Two commonly used metricsare:

- Perplexity
- ExactMatch@N

## Perplexity

Perplexity[27] is a standard metric used extensively in the offline evaluation of language models. This metric measures how accurately the model predicts the exact sequence of tokens present in text data. Inmathematical terms,perplexity is defined as the exponential of the average"negative log-likelihood"of the predicted probability given the previous tokens in a sequence:

<!-- formula-not-decoded -->

In this equation:

- accurately the model predicts the sequence.
- N is thenumber of tokens in the sequence.
- tokens,

Figure 2.38 illustrates a concrete example to better understand perplexity.

Figure2.38:Example of perplexity calculation

<!-- image -->

A lower perplexity value indicates that the model has assigned higher probabilities,on average,to the tokens that appear in the text data.Therefore,a lower perplexity means the model isbetter atpredicting the next tokens.

## ExactMatch@N

ExactMatch@N measures the percentage of generated phrases that are exactly N words long and that match the first N words of the ground-truth text.Figure 2.39 shows ExactMatch@3 calculations for three generated sequences. In practice,there are usually more than three sequences to evaluate, Calculating ExactMatch@N for differentvaluesof N allows us to measure how themodel performs at different suggestion lengths. To measure the overall performance of the model,we calculate the ExactMatch for all lengths up to a specific length and then take the average.

Figure 2.39:Example of calculating ExactMatch@3 for three sequences

<!-- image -->

WhilePerplexity and ExactMatch@N have traditionally been used to evaluate Gmail Smart Compose,other metrics such as BLEU score and ROuGE-N,introduced more recently have been found to be helpful. We examine these metrics in more detail in Chapter 3.

## Online evaluation metrics

engagement, the model's latency,and the overall impact on user experience.

ones,In this section,we focus on the following metrics:

- User engagement metrics
- Effectiveness metrics
- Latency metrics
- Quality metrics

## User engagement metrics

- ·Acceptance rate:The percentage of suggestions made by the Smart Compose feature that are accepted by users.A higher acceptance rate indicates that the suggestions are relevant and useful to users.
- .Usage rate:The percentage of all composed emails that have utilized the Smart Compose feature. High usage rates typically indicate that users trust the feature.

## Effectiveness metrics

- ·Average completion time: Tracks the average time taken by users to compose emails with and without the aid of Smart Compose. A reduced average completion time using Smart Compose will indicate that the feature is speeding up the email writing process.

## Latency metrics

- ·System response time: Measures the time it takes for the Smart Compose suggestions to appear after the user begins typing. It's important to ensure this metric stays below a certain threshold so the suggestions are made before the user types them.

## Quality metrics

- ·Feedbackrate:Measures the rate at which users provide feedback on the suggestions. Feedback ishelpful for continuous improvement of the system.
- Human evaluation:Qualitative assessments through user studies are employed to evaluate the usefulness of suggestions.This metric reflects user satisfaction with the Smart Compose feature.

These online metrics are essential for evaluating how well Smart Compose feature works in production. By monitoring these metrics, the stakeholders can obtain a holistic view of feature'sperformance.

## Overall ML System Design

In this section, we propose a design for a simplified Smart Compose feature.

When designing such a feature,we should consider more than just the underlying model that predicts the next token.The system's effectiveness depends on various components working together to ensure the system isresponsive,generatesrelevant suggestions,and maintains ethical standards. For the Smart Compose feature, we examine the following key components:

- ·Triggering service
- ·Phrase generator
- ·Post-processing service

Let'sexplore each inmore detail.

## Triggering service

Smart Compose, as the additional context allows for more useful suggestions.

generator component,which we discuss next.

## Phrase generator

likely completion based on thepartial text the user has already typed.

To achieve this, the phrase generator interacts with the trained model and employs beam search to generate the top-k most probable completions. Each completion ends with the (Eos) token and an associated score that indicates how confident the model is about the completion.

Figure 2.40: Beam search outputs top five potential completions (beam width = 5)

<!-- image -->

Given the possible completions,two critical considerations are necessary:

- Removing long suggestions
- Removing low-confidence suggestions

## Removing long suggestions

Figure 2.41:Removing long suggestions

<!-- image -->

## Removing low-confidence suggestions

We remove suggestions with confidence scores below a certain threshold. This ensures we do not present suggestions if the model is not confident enough about it.

Figure 2.42: Removing low-confidence suggestions

<!-- image -->

Finally,if the final list of suggestions isnot empty,the phrase generator will pass the one with the highestconfidence score to thepost-processingservice.

## Post-processing service

The post-processing service addresses potential biases before suggestions are presented to the user. This component follows predefined rules to detect and correct bias effciently. Common strategies to achieve this include:

- ·Pronoun replacement: Replace gender-specific pronouns to ensure neutrality. For example,"he" or "she" might be replaced with "they" in contexts where gender is not specified,
- ternatives where appropriate. This includes changing words like "chairman" to "chair-

person"or"policeman" to"police officer."

- as respectful and neutral.
- phrases,and patterns to detect and remove problematic content.

respectful,and inclusive.

Figure 2.43:Smart Compose feature overall design

<!-- image -->

Compose feature:

- 1.Monitoring: The triggering service monitors the user's activity as they type.
- terns,
- pletions from the trained model.
- suggestions and those with low confidence scores.

sensitive terms.

## Other Talking Points

some topics you should prepare for:

- Supporting Smart Compose in multiple languages [28].
- ·Personalizing suggestions[28].
- Incorporating additional context for better predictions [28].
- Understanding how different tokenization algorithms work, such as BPE [11], SentencePiece [12], and WordPiece [29].
- ·Understanding different ML objectives such as masked language modeling (MLM) and its variations [18].
- ·The multi-token prediction objective and its pros and cons [30].
- ·Balancing quality and inference latency [28].

## Summary

<!-- image -->

## Google Translate

## Introduction

Google Translate is a widely used language translation service offered by Google. The servicerelies on machine learning (ML) models to understand and translate text between languages. As of 2024, the service supports over 130 different languages and has over a billion users[1]. This chapter explores the system design behind a language translation service.

Figure 3.1:Language translation service

<!-- image -->

## Clarifying Requirements

Here is a typical interaction between a candidate and an interviewer:

Candidate: Are there specific languages the system should support initially? Interviewer:Let'sfocuson four languages:English,Spanish,Korean,and French.We can expand tomorelanguagesinthe future.

Candidate: Considering the diversity of languages, do we have access to a suffciently large and varied dataset for training?

Interviewer: Yes. We have access to a substantial multilingual corpus, including formal documentswebcontent,and conversational texts,inall four languages.hedatasetcon tains 3oo million examples,an example being defined as a pair of sentences in the source and target languages.

Candidate: Do we have access to general text data? This is important,as it would allow guages,obtained from various sources.

tomatically?

automatically.

Candidate: Is there a constraint on the length of the input text?

words.

In other words,should the model work on-device?

will be deployed on the cloud.

Candidate: Should the system support real-time translation?

Interviewer: Not initially.

## Frame the Problem as an ML Task

In this section,we frame the problem of building a translation system as an ML task.This involves understanding the system's inputs and outputs and choosing a suitable ML approach.

## Specifying the system's input and output

The input to a translation system is a sequence of words in the source language and the target language provided by the user. The output is a sequence of words in the target language.

Figure 3.2:Input and output of a translation system

<!-- image -->

## Choosing a suitable ML approach

In language translation, a sequence of words in one language is transformed into a sequence of words in another language. This sequence-to-sequence (seq2seq) structure is also found in other tasks,such as text summarization and speech recognition.

Seq2seq models,a class of ML models,are specifically designed to handle such tasks. These models transform an input sequence into an output sequence,which can vary in length from the input.Seq2seq models follow an encoder-decoder architecture,which has twomain components:

- Encoder:Processes the input sequence and transforms it into a sequence of context vectors,thus encoding the information in the input sequence.
- ·Decoder: Utilizes the encoder's context vectors to generate the output sequence one token at a time.

<!-- image -->

originally introduced in the context of language translation[2].

As described in Chapter 2,Transformers have three variations:encoder-only,decoderonly, and encoder-decoder architecture. Encoder-only models such as BERT [3] are good at understanding and processing the input sequence but typically require additional mechanisms to generate output. Decoder-only models, such as OpenAl's GPT [4] and Anthropic's Claude[5],arehighly effective atgenerative tasks.

While all Transformer variations offer strong performance and can be adapted for language translation tasks through techniques like prompt engineering,encoder-decoder models are usually preferred for three main reasons.First,the encoder-decoder architecture separates the input understanding and output generation,which is ideal for seq2seq tasks such as language translation. It allows the encoder to specialize in the source language and fully understand the input sequence before the decoder generates the output. For example,encoders often use bidirectional mechanisms such as bidirectional STMs6or Transformers,which enable them to understand the context from both directions.

inputs and outputs do not have a fixed length relationship.

section,

## Data Preparation

In thissection,wepreparerawtextual datafortheencoder-decoderTransformerWehave pairs,achcontainingasource-language sentence and its corresponding translation in the targetlanguage.

Raw text in both general data and translation data is often noisy and not in the format the ML model expects. Since we covered preparing general data in Chapter 2, we willfocus on preparing translation data. In particular, we focus on the following two steps:

- 1.Text preprocessing
- 2.Text tokenization

## Textpreprocessing

We apply thefollowingpreprocessing techniques toraw text in translation data:

- ·Remove missing data:Remove pairs where either the source or target text is missing
- ·Remove noisy data:Remove pairs with HTML tags or incorrect language pairings.
- ·Deduplication:Remove duplicate sentencepairs from the dataset to prevent the model fromoverfittingtocertain examples.
- Handle named entities:Language translation models often struggle with named entities.We identify these entities in the text and replace them withplaceholder tokens.After translation,we replace the tokens with the original entities.For example, consider the sentence:"The California city,Burlingame, is named after diplomat Anson Burlingame." First, we detect the named entities:"California (location name)," "Burlingame (location name)"and "Anson Burlingame (person's name)" Next,we replace these entitieswith placeholder tokens:"The ENTITY\_1 city,ENTITY\_2,isnamed after diplomat ENTiTY\_3.This approach helps the model focus on the sentence context during training without being confused by uncommon terms.

In modern language translation,particularly with models like Transformers,some tradi tional preprocessing steps have become less crucial or are handled differently.Here are a few preprocessing steps that were once essential in traditional translation models but are now unnecessary or lessrelevant:

- ·Lowercasing: Modern language translation models can handle case sensitivity as part of their training.They can learn to distinguish between different forms of words based on case (e.g.,"Apple" as a company vs."apple" as a fruit) without needing to convert everything to lowercase, Therefore, lowercasing is often skipped to preserve the original case information,
- ·Stop word removal: Stop words (e.g, "the""and," "in") are essential for the gram-

natural translations.

- move valuable information.

## Text tokenization

our vocabulary,which is huge and inefficient.

<!-- image -->

| Token       |   ID |
|-------------|------|
| <BOS>       |    0 |
| <EOS>       |    1 |
| walking     |    2 |
| bonjour     |    3 |
| hello       |    4 |
| fantastique |    5 |

Word-level Vocabulary

Figure 3.4:A huge vocabulary size due to word-level tokenization

88| Chapter 3.Google Translate to examine Byte-Pair Encoding(BPE)[7],a commonly used subword-level tokenization algorithm,in detail.

## Byte-Pair Encoding(BPE)

BPE builds a subword-level vocabulary through iterativemerging.It startswith characters and iteratively merges the most frequent combinations into new subwords.This allows themodel tobreak down words,evenrare or unseen ones,intoknown components,thus enabling accurate understanding and translation.Let'swalk through a concrete example tobetter understandBPE.

## Initial setup

Suppose we have a corpus with the following set of words:"cat，""cats,""dog"and"dogs."

Figure 3.5:Frequency of words in our corpus

<!-- image -->

In the initial setup,our goal is to initialize a vocabulary consisting of different characters and their frequency of occurrence in the corpus.To achieve this,we follow these steps:

- 1.Add a special end token,"&lt;/w&gt;,"to the end of each word to mark its boundary. This special tokenhelpsthe model knowwhen aword has ended.
- 2.Tokenize the corpus by breaking down each word into individual characters.
- 3.Initialize the vocabulary with individual characters and their frequency of occurrence.

|   # |      |   Token Frequency |
|-----|------|-------------------|
|   0 | C    |                 8 |
|   1 | a    |                 8 |
|   2 | t    |                 8 |
|   3 | S    |                 7 |
|   4 | d    |                10 |
|   5 | 0    |                10 |
|   6 | g    |                10 |
|   7 | </W> |                18 |

<!-- image -->

Initial vocabulary

Figure 3.6:Initial setup steps

## Iterative merging

size or meets the stopping criteria.

## Following are the first five BPE iterations:

Figure 3.7:BPE iteration 1

<!-- image -->

|   # | Token   | Frequency   |
|-----|---------|-------------|
|   0 | C       | 8           |
|   1 | a       | 8           |
|   2 | t       | 8           |
|   3 | S       | 7           |
|   4 | d       | 10-10=0     |
|   5 | 0       | 10-10=0     |
|   6 | g       | 10          |
|   7 | </w>    | 18          |
|   8 | op      | 10          |

- ·Iteration 2:Now we look for the next most frequent pair,which is "do and "g(also to create the token"dog"
- be represented as"ca" and "t" The token "ca" now has a frequency of 8.

Figure 3.8:BPE iteration 2

<!-- image -->

|   # | Token   | Frequency   |
|-----|---------|-------------|
|   0 | C       | 8           |
|   1 | a       | 8           |
|   2 | t       | 8           |
|   3 | S       | 7           |
|   6 | g       | 10-10=0     |
|   7 | </w>    | 18          |
|   8 | do      | 10-10=0     |
|   9 | bop     | 10          |

- Iteration4:We continue by merging"ca"and "t", which appear 8 times together (from "cat" and "cats"). We merge them to form the token"cat."Now,"cats" can be represented as"cat" and "s"", and "dogs" as"dog" and "s"The frequency count for"cat" is updated to8.
- Iteration 5:Finally,the next most frequent pairing is"s" and "/w&gt;"(from"dogs" and "cats"),which appears 7 times.We merge"s"and"&lt;/w&gt;"to form the token"s&lt;/w&gt;""

|   # | Token   | Frequency   |
|-----|---------|-------------|
|   7 | </w>    | 18-7=11     |
|   9 | dog     | 10          |
|  11 | cat     | 8           |
|  12 | s</w>   | 8           |

Figure 3.9:BPEiterations3-5

<!-- image -->

BPEiterativelymerges themostfrequentcharacterpairs,leadingtoa morecompactrepresentation of the corpus. The merging continues until we reach the desired number of tokens oriterations.

Figure 3.10:BPEvocabulary after 5 iterations

|   # | Token   |   Frequency |
|-----|---------|-------------|
|   7 | </w>    |          11 |
|   9 | 6op     |          10 |
|  11 | cat     |           8 |
|  12 | s</w>   |           8 |

forms. For instance, the token "cat" followed by"&lt;/w&gt;" indicates the end of the word "cat,whereas the token "cat" without &lt;/w&gt; could also be part of another word. This distinction helps BPErepresent and interpretwords accurately during translation,allowing it to efficiently handle both familiar and unseen words.

Once the vocabulary has been created,we construct our training data byreplacing each tokenized sentence with a sequence of integers.This leads to multiple tables,each for a specific language pair.Figure 3.11 shows the prepared translation data for the EnglishFrench and English-Korean language pairs.

Figure 3.11:Constructed training data for English-Korean and English-French pairs

<!-- image -->

## Model Development

We utilized an encoder-decoder Transformer to train language translation. In this section, we explore the architecture of the encoder and decoder,training strategies,and sampling methods.

## Architecture

and decoder separately and highlight theirkey differences.

## Encoder

input token.

Figure 3.12:Encoder components

<!-- image -->

The encoder consists of the following components:

- Text embedding
- Positional encoding
- Transformer

Text embedding: This component converts each input token into an embedding vector. These embeddings capture the semantic information of each token.

Figure 3.13:Token embedding table

<!-- image -->

| Token   | Embedding   |
|---------|-------------|
| </W>    |             |
| 6op     |             |
| cat     |             |
| s</w>   |             |

Positional encoding: The positional encoding component injects information about the position of each token in the input sequence. As discussed in the previous chapter,both fixed and learned methods are effective in practice.For simplicity,we choose a fixed po- sitional encodingmethod such as sine-cosine encoding.

handle this sequence length without significant performance issues.

## Decoder

put and the previously generated tokens.The decoder has the following components:

- Text embedding: Converts each token in the target sequence to an embedding
- Positional encoding: Injects information about the position of each token
- Transformer:Processes the target sequence and outputs an updated sequence of embeddings
- Prediction head:Utilizes the updated embeddings to predict the next token.

Figure 3.14:Decoder components

<!-- image -->

## Whatarethekeydifferencesbetween the encoderand decoder?

There are threekey differencesbetween the encoder and decoder:

- Cross-attentionlayer
- ·Self-attentionmechanism
- ·Predictionhead

## Cross-attentionlayer

The Transformer component in the decoder includes a cross-attention layer. This layer performs the MHA mechanism over the encoder's output. It enables each token in the decoder to attend to all embeddings in the encoder. This allows the cross-attention to effectively integrate information from the input sequence during the generation of the outputsequence.

Figure 3.15:Cross-attention layer

<!-- image -->

Self-attention mechanism

Figure 3.16:Different self-attention mechanisms in the encoder and decoder

<!-- image -->

## Prediction head

The decoder has a prediction head on top of the Transformer component.The prediction head usually includes a linear layer followed by a softmax layer to convert the Transformer's output into probabilities over the vocabulary. These probabilities are used to determine the most likely next token.

## Training

- 1.Unsupervised pretraining
- 2.Supervised finetuning

## 1.Unsupervised pretraining

base model capable of understanding language,grammar,and context.

## Pretraining data

We utilizepopular pretraining datasets suchasC4[8]Wikipedia[9],andStackExchange [10]. In contrast to Chapter 2, where we focused on pretraining a language model for Englishonly,forlanguage translationweneedabasemodewithageneralunderstanding of multiple languages.Therefore,we do not remove non-English text data from these remove any text data belonging to languages outside that set.

## MLobjective and loss function

In Chapter2,we explored next-tokenprediction as theprimaryMLobjectivefor language generation. Next-token prediction is not an ideal choice in encoder-decoder pretraining because the training is unsupervised. If we pass an entire sentence to the encoder, it will encode information in a way that allows the decoder to always predict the next word accurately, thus effectively"cheating"Instead,we use"masked language modeling"a common ML objective for pretraining an encoder-decoder Transformer. Let's examine it in more detail.

## Masked language modeling(MLM)

In MLM, also known as masked token prediction, some of the input tokens are masked, and themodel is trained topredictthosemasked tokens.

Figure 3.17:An overview of the MLM objective

<!-- image -->

MLM allows the encoder to process the input sentence and encode it so the decoder can predict the masked words. As the masked words are never visible during the encoding process,this prevents the model from cheating,

Tomeasure themodel's performance in predicting the masked tokens,we use cross-entropy loss. This commonly used loss function measures the discrepancies between the predicted probabilities and the ground-truth tokens,thus guiding the training process.Here isa stepby-step explanation of how loss is calculated using the MLM objective：

- might become"Thank[MASK] for inviting[MASK]"
- token.
- input sequence during training.
- 4.The decoder predicts the next token for each position in the sequence.Each prediction uses all previous input tokens and encoded information from the encoder.
- 5.Calculate the cross-entropy loss over the predicted probabilities and the ground-truth for the masked tokens only.

Figure 3.18: Cross-entropy loss calculation for MLM objective

<!-- image -->

98|Chapter 3.Google Translate process this encoded information and predict themasked tokens.This objective prepares both the encoder and decoder for the supervised finetuning stage.

resource-intensive and,therefore,expensiveInpractice,weoftenuse publicly available encoder-decodermodels such as Google's T5[12] orMeta'sBART[13],which havebeen pretrained on extensive datasets. This approach significantly reduces the cost and resources needed for pretraining.

## 2. Supervised finetuning

Supervised finetuning,the second stage of our training process,adapts thebase model to the specific task of language translation.It does this by finetuning the base model on translation data.To adapt the base model to language translation,wehave two options:

- 1.Bilingual approach
- 2.Multilingual approach

Option 1:Bilingual models

Option2:Multilingualmodel

Figure 3.19:Bilingual vs.multilingual models

<!-- image -->

## Bilingual approach

In this approach,we train models specific to each language pair.Training language-specific models has several advantages. First,they capture the unique linguistic nuances of each language pair.Second,they usually demonstrate higher translation accuracy due to their specializednaturesFinally,improvingperformanceissimplerwith language-specifcmod els because we can easily isolate and address specific issues that may arise for each language pair.However, training, deploying,and maintaining multiple models is resourceintensive and costly.

Multilingual approach a bilingual approach.

## Training data

translation in the target language.

English-Korean tokenized pairs

| English                    | Korean                                 |
|----------------------------|----------------------------------------|
| [138,18,9,2130]            | [186,732,666,349,818]                  |
| [138,9561,31, 721]         | [226,91022,82483, 9643,40344]          |
| [309,11001,22, 70701,3752] | [39485,128320,8532, 4432,54255,196710] |

English

[15724,374,9439]

[2028,374,15526]

[4438,527,499, French

[12319,20272,282,

1607,75804,88253

[34,17771,1377,57332]

[10906,44496,2442.84

30]

30]

English-French tokenized pairs

Figure 3.20:Example ofprepared training data for different language pairs

## ML objective and loss function

Whereas the pretraining stagewas unsupervised,the finetuning stage is supervisedThe generates the target sentence tokens. Since the decoder should generate tokens sequen tially after training,we use next-token prediction as our ML objective. We use cross entropy as our loss function to measure the accuracy of the predicted next token.

Figure 3.21:Loss calculation during the finetuning stage

<!-- image -->

Figure 3.21 shows loss calculation during the finetuning stage.For simplicity,it visualizes a single prediction. In practice,as we saw in Chapter 2,the decoder predicts the next token for all positions simultaneously,and the losses are calculated for all predictions.

## Sampling

During sampling, the trained model generates a potential translation by predicting each subsequent token based on the previously generated tokens and the context of the input sequence, As discussed in Chapter 2,there are two main strategies for sampling text in generative models: deterministic methods (e.g.,beam search) and stochastic sampling. Here,we choose beam search for two main reasons:

Figure 3.22:Generating translation

<!-- image -->

1. Translation accuracy: Beam search usually leads to more accurate translations. This is because the algorithm evaluates multiple possible sequences and selects the most probableone.
2. 2.Consistency:Beam search is deterministic,meaning it always produces the same out put given the same input.This consistency ensures that translations will provide few surpriseswhich is critical in most translation systems.While diversity can be benef cial,it is neither essential nor desirable for language translation systems.

stochastic methods such as top-k and top-p sampling in detail.

Table 3.1:Comparison of deterministic and stochasticmethods

| Characteristic   | Deterministicmethods                                              | Stochastic methods                                                    |
|------------------|-------------------------------------------------------------------|-----------------------------------------------------------------------|
| Approach         | Followapredictableprocess togenerate output                       | Generate outputbased onprobability distribution                       |
| Efficiency       | Typicallylesseficientdue to tracking multiplepaths                | More efficient sincerandomnessallowsfor quicker selections            |
| Quality          | Coherent and predictable                                          | Diverse and creative                                                  |
| Risk             | Usuallyleadtorepetitiveoutputforlonger sequences                  | Might produceinappropriate output due to their creativeness           |
| Use case         | Suitablefortasksrequiringconsistency, such aslanguage translation | Suitablefortasksrequiringcreativity,such as open-ended textgeneration |
| Methods          | Greedysearch,beamsearch                                           | Multinomial,top-k,top-p                                               |

## Evaluation

## Offline evaluation metrics

To thoroughly evaluate a language translation model,metrics should measure both translation accuracy and contextual appropriateness. The research community has proposed several metrics that,over the years,have become widely accepted as standards. Some commonly used metrics are:

- BLEU
- ROUGE
- METEOR

## BLEU

BLEU (BiLingual Evaluation Understudy) [16] is a precision-based metric that compares ngrams (a sequence of "n"words) of the candidate translation with n-grams of the reference translations and counts the ratio of matches. It ranges from O to 1,where a higher value indicates a more precise translation.

The BLEU score is calculated using the following formula:

<!-- formula-not-decoded -->

## where:

- ·N is the maximum n-gram length considered for evaluation
- ·BP is the brevity penalty

- Pn is the n-grams precision
- ·w represents the weight for different n-gram precisions

Let'sexplore each of these terms in detail.

## BrevityPenalty(BP)

The formula is:

where:

- cis the translation length
- ·r is the reference translationlength

If the candidate translationlength,c,sgreater than thereference translationlength,r,the brevity penalty is 1 (ie.,there is no penalty).If the candidate translation length is less than or equal to the reference translation length, the brevity penalty is an exponential decay based on the ratio of the lengths.

## Precision(pn)

Precision measures how many n-grams in the candidate translation are present in reference translations. It is calculated by dividing the number of matching n-grams by the total number of n-grams in the candidate translation.Figure 3.23 provides an example of calculating p2 for a candidate and one reference sentence.

<!-- formula-not-decoded -->

Reference:The sun sets over the calm sea.

Candidate:The sun sets over the peaceful sea

Figure 3.23:Example of calculating precision for 2-grams(p2)

<!-- image -->

## Weights(wn)

These weights correspond to the precision of each n-gram size. Usually, we distribute them evenly,giving each n-gram precision the same importance.Forinstance,forn-grams up to4-grams,eachwwouldbe1/4.

BLEU's main advantage is that it is simple and easy to compute. However, it has a major drawback: it can unfairly penalize translations that are correct but different from the reference translation.For example,if the reference translation is"The engineer discovered a new algorithm, and the generated translation is"The engineer found a new method BLEU might penalize the generated translation even though it conveys the same meaning. Despite this limitation,BLEU remains insightful and widely used in practice to evaluate language translation models.

## ROUGE

ROUGE(Recall-Oriented Understudy for Gisting Evaluation)[17] is a popular metric that complements BLEU by focusing on recall instead of precision. It measures the ratio of n-gram overlaps between the candidate and reference texts.For example,the ROGUE-N recall is defined as:

Recall=

Number of matching n-grams

Total number ofn-grams in the reference

If you are interested to learn more about ROuGE and its formula,refer to[17].

semantically similarwords mightreceive a low RouGE score.

## METEOR

then combines these measurements using a weighted harmonicmean.

These synonyms are found using linguistic resources suchas synonym dictionaries orlexi cal databases.One commonly used resource isWordNet [19],which organizeswords int synonyms of various types and shows the relationships between those synsets.

Figure 3.24:Example of relationships between words

<!-- image -->

at its pros and cons,

## Pros:

- and stemming during the evaluation of translations.
- plete.
- ments than BLEU and ROUGE,

106 | Chapter 3.Google Translate

## Cons:

- ·Computational complexity:METEOR is harder to implement and takesmore time to synonym and stemmingmatching.
- ·Resource dependence: METEOR relies on linguistic resources such as synonym dictionaries and stemming algorithms,whichmay notbe available forall languages.

To summarize,all three metrics offer insights into the model's performance and are commonly used in practice. Let's transition to online evaluation to understand how our model performs in real-world scenarios.

## Online evaluation metrics

During the online evaluation, we evaluate how well our language translation system works in production. We use the following two metrics to measure how satisfied and engaged ourusers are:

- User feedback:Collect ratings or feedback from usersregarding the quality of translations. The metric is insightful since it directly reflects user satisfaction.
- ·User engagement:Measure users'engagement by monitoring how often they use the translation feature,how long they interact with it,and how frequently they return. This helps us understand how valuable and effective the translation tool is in realworld use.

Figure 3.25:Collecting user feedback

<!-- image -->

Combining offline and online evaluation metrics gives us a more complete view of the technical standards and satisfy user expectations.

## Overall ML System Design

we examine twokey components:

- Language detector
- ·Translation service

## Language detector

classification task,and encoder-only architecture is a good candidatearchitecture forsuch a task. We can modify the encoder-only Transformer in two ways (Figure 3.26) to clasify input sentences:

- 1.Averagepooling:Pass the Transformer's outputs to an average pooling layer,and then a prediction head to output language class probabilities.
2. Last token representation: Use the last token representation from the Transformer's output and feed it to the prediction head forprobability prediction.

Figure 3.26:Two options for building a language detector using an encoder-only Transformer

<!-- image -->

## Translationservice

The translation service interactswith the specificmodel based on thedetected and desired languages. It then applies beam search to generate a sequence of tokens in the target language and converts the tokens back into text. The final translation is then shown to the user.

Figure 3.27: Language translation overall design

<!-- image -->

## Other Talking Points

If time permits at the end of the interview,consider discussing these additional topics:

- Supporting translation for languages with limited training data using transfer learning and multilingual models[20].
- ·Approaching language translation using a decoder-only Transformer [21].
- Continuously improving translation models through user feedback[22].
- Optimizing techniques for efficient inference and on-device translation[23].
- Developing a single multilingual model [24].
- ·Other automatic metrics such as WER and how they are calculated [25,26].
- ·How to build a language detectionmodel [27].

<!-- image -->

## ChatGPT:PersonalAssistant Chatbot

## Introduction

ChatGPT [1] is a chatbot that was developed byOpenAl and launched in 2022. It generates human-like text based on the input it receives. The chatbot can assist with various tasks, including answering questions,providing explanations,andproducingcreative content.

ChatGPT has quickly become one of the most adopted applications in history. It gained over 100 million users in less than three months after its launch [2]. This rapid growth highlights the capabilitiesof generative Al and itspotential to helpwith day-to-day tasks and enhance productivity. In this chapter,we examine the key components of building a chatbot similar to ChatGPT.

Figure 4.1:Example of a conversation with ChatGPT

<!-- image -->

## ClarifyingRequirements

Here isa typical interaction betweena candidate and an interviewer:

Candidate:What languages should the chatbot support?

Interviewer:Initially,let'sfocusonEnglish.

strict content moderation and algorithms in place. Is that a fair assumption? Interviewer:Certainly.

answering questions.

orvideo?

text.

maintain the conversation context?

within the same conversation session. Let's say it is expected to have a context window of atleast4096tokens.

Candidate: Is the chatbot expected to be able to browse websites, call external APl, or search online?

Interviewer: Let's not focus on that in this round.

Candidate:Should the chatbot personalize its interactionswith users?

Interviewer:Let's notfocus onpersonalization.

Candidate:Do we have instruction-based training data? Interviewer:Yes,wehave a datasetwith 80o0 examplesof instructions and answers.

## Frame the Problem as an ML Task

## Specifying the system's input and output

contextually appropriate response generated by the chatbot.

Figure4.2:Inputandoutputofa chatbot

<!-- image -->

## Choosing a suitableML approach

Developing a chatbot is a textgeneration task inwhich a language model processesan input prompt and generates a response. This language model typically needs billions of parameters to learn effectively; hence, they are often called large language models (LLMs).

As we saw inpreviouschapters,thedecoder-onlyTransformer is thestandard architectural choicein language models.Mostmodern LLMs,such asOpenAl'sGPT[3],Google's Gemini [4],and Meta's Llama[5],are all based on the decoder-only Transformer.In line with these models,we use a decoder-only Transformer tobuild our chatbot.

## Data Preparation

The effectiveness of an LLM depends on the quality of its training data, which mainly comes from web sources.This data,often automatically crawled from websites,forums, and blogs,requires special considerations and careful preparation. The most common steps include:

- Content extraction and parsing: Web-crawled data often contains extraneous elements,for example,HTML tags,advertisements,and navigation links.This step nvolves parsing the raw HTML content using libraries such as Beautiful Soup [6] or Ixml [7] and extracting the main text body while discarding irrelevantsections. Techniques such as DoM analysis[8] and boilerplate detection[9] are applied to isolate and retain the core content relevant to language modeling.
- URL and domain filtering:Not all web domains provide high-quality or relevant content. URL filtering uses predefined rules or machine learning (ML) classifiers to exclude unwanted sources,for example,low-quality blogs,content farms,or spam sites.Do main allowlistingorblocklisting techniques are also applied tocurate data from trusted and relevant sources,ensuring the dataset's quality and reliability.
- Languageidentification:Crawled data often includesmultilingual content,whichneeds to be filtered to match the target language(s) for training. Language detection tools such as fastText[10] or langid.py[11] are employed to classify and filter documents.
- ·Content quality filtering: Not all web content is equally valuable for training purposes. Quality assessment techniques,including readability scoring,spam detection algorithms,and heuristiccheckse.g,content length,sentence structure),areused o evaluate and filter low-quality text.ML models can also be employed to predict the quality of web content based on features extracted from the text.This step is crucial to ensure that only high-quality data is used for training.

- safecontent.
- guidelines.
- trainingdata ishigh in quality and useful for training.
- websites, we identify and retain only one copy. This step reduces redundancy in the training dataset and ensures themodel is not overexposed to certain data.
- ·Remove irrelevant data: We use heuristics and rule-based methods to remove irrelevant data. For example, we remove texts with non-standard characters or in languages that the chatbot isnotexpected to support.
- Tokenize text:We use a subword tokenization algorithm such as Byte-Pair Encoding (BPE) to tokenize the text data.To review BPE,refer to Chapter3.

## Model Development

## Architecture

The LLM's architecture is based on a decoder-only Transformer. While the text embedding

## Where:

- ·qm is the query vector atposition m,

Figure 4.3:Components of a decoder-only Transformer

<!-- image -->

Let's examine LLM's positional encoding in more detail.

## Positional encoding

In a chatbot setting,theinputsequence is typically much longer than a single sentence or email.Asper the interviewer'srequirements,ourgoal is to build a system with a context window of at least 4096 tokens.This requires a positional encoding method that allows the model to understand the positions of all the tokens and the relationshipsbetween them.

In thissection,webegin with a briefreview of absolutepositional encoding followed by an exploration of relative positional encoding.Finally,we delve into rotary positional embedding (RoPE) [12], a robust positional encoding method used by popular LLMs such as Llama 3 [13].

## Absolutepositional encoding

Absolute positional encodingrefers to traditional methods such as sinusoidal or learnable encodings whereby each position in a sequence is represented by a unique vector.

In this approach,encoded positions are then added to the token embeddings,providing the model with information about where each token appears in the sequence. Formally, the attention keys and queries are computed using the following equations:

<!-- formula-not-decoded -->

- ·k, is thekeyvector atpositionn,
- WandWkarelearnableweightmatrices,
- emand enare token embeddings atpositionsm andn,

<!-- formula-not-decoded -->

sinusoidal patterns tend to become repetitive over long distances,resulting in a loss of information about token relationships. This shortcoming is addressed by relative positional encoding.

## Relative positional encoding

In relative positional encoding, instead of encoding the absolute positions of tokens, we encode the differences in the positions of two tokens. This way, the model can focus on the relative distancesbetween tokens,which is often more important than their abso lute positions.For example,in a sentence,knowing that the word"car" follows the word "chased" is more informative than knowing the positions of the tokens as numbers 5 and 10.

The attention calculation in relative positional encoding can be expressed in different ways. The T5 paper [14] suggests that the second and the third interaction terms in the original expression in absolute positional encoding can be dropped and that the fourth term could be replaced by a learnable bias:

<!-- formula-not-decoded -->

relative positional vector Rn-m:

<!-- formula-not-decoded -->

RoPE,whichwe examine next,addresses thislimitationbyencoding bothabsoluteand relative positional information through a rotation in the embedding space.

RoPE represents positional information as a rotationmatrix applied to the token embeddings.This can be described mathematically as follows: Given an input sequence,RoPE applies a rotation matrix to each embedding. This transformation can be expressed as:

## Rotary positional encoding(RoPE)

<!-- formula-not-decoded -->

where qm is token embedding at the position m,andR(em) is a rotation matrixparameterized by the positional angle Om.This angle is typically derived from the position indexm and is constructed in such a way that therotation capturesboth the absolute andrelative position information.This rotation matrix,constructed using trigonometric functions,rotates the embeddings in thecomplexplane,capturing both absolute and relative positional information.

Figure 4.4:RoPEin2D

<!-- image -->

Figure 4.4 shows how rotational position encoding works by rotating word embeddings in a two-dimensional space. The words"cat" and "dog" are represented as vectors, and the angle between them,denoted by θ,encodes their positional relationship.On the left, the sentence is "The cat chased the dog." The position of"cat" is shown in red, and the position of "dog" is shown in blue.The angle between these vectors captures the relative positioning of these two words in the sentence.

On the right, another sentence"Once upon a time, the cat chased the dog" is shown. Notice that the relative angle θ between the"cat" and"dog"vectors is still the same,but their absolute positions are different. This demonstrates how RoPE captures both the absolute and relativepositions of words,allowing the model to understand theorder and distance between words in a sentence.

In a higher-dimensional space, the rotation matrix can be extended to accommodate d dimensions:

## Pros:

- changes inpositionbetter than other methods.
- which encode position additivelywithout leveraging thisgeometric insight.
- ·Generalization to unseen positions: Because RoPE encodes position through rotations, the resulting embeddings maintain consistent relationships,regardless of abso lute position.This allows forbetter generalization across varying sequence lengths.

## Cons:

- Mathematical complexity:RoPE introduces additional mathematical operations involving rotations in the embedding space.While these are not overly complex,they are more intricate than traditional positional encoding methods such as sinusoidal or learnedpositional embeddings.

## Training

including ChatGPT, use a three-stage training strategy:

- Pretraining
- Supervised finetuning (SFT)
- Reinforcement learning from human feedback(RLHF)

Let's discuss each stage in more detail to understand its purpose.

<!-- formula-not-decoded -->

Figure 4.6:Three stages of training an LLM

<!-- image -->

## 1. Pretraining

Pretraining is the initial stage of the training process. In this stage, a model is trained with an enormous volume of text data from the Internet.Thepurposeofpretrainingis to create a base model with a broad understanding of language and world knowledge.

The pretraining stage requires significant computational resources. It typically requires thousands of GPus,costs millions of dollars, and takes months of training.

## Pretraining data

The pretraining data typically consists of a large corpus of general text data from various sources on the Internet,forexample,webpages,books,and social media posts.

Several datasets are commonly used when pretraining LLMs. Each serves a unique purpose, from broadening the model's exposure to diverse language styles to deepening its understanding of specific domains. Commonly used datasets are:

- Common Crawl: Common Crawl [17] is a publicly available dataset collected from a large number of web pages on the internet.It contains petabytes of data that have been regularly collected since 2008. This data often includes irrelevant information and harmful content;hence,ignificant datacleaning isneeded tomakeit suitableor training LLMs.
- ·C4:C4[18], created by Google, is a cleaned version of the Common Crawl dataset specifically used for training LLMs,
- GitHub:The GitHub dataset comprises a vast collection of open-source code reposi tories. Its purpose is to help the model understand programming languages and code structures.

Model Development | 123

- writtenandeditedmorecarefully.
- performanceof LLMs.
- demicdomain.
- swers,primarilyintheformatofadialoguebetweenusers.

4.1 shows theproportion of each dataset used byLlama1 during training.

Table 4.1:Llama 1 pretraining dataset

| Dataset       | Sampling proportion   | Disksize   |
|---------------|-----------------------|------------|
| Common Crawl  | 67.0%                 | 3.3TB      |
| C4            | 15.0%                 | 783GB      |
| Github        | 4.5%                  | 328GB      |
| Books         | 4.5%                  | 85GB       |
| Wikipedia     | 4.5%                  | 83GB       |
| ArXiv         | 2.5%                  | 92GB       |
| StackExchange | 2.0%                  | 78GB       |

## ML objective and loss function

kens,

## The outcome of the pretraining stage

ating relevant and meaningful text.

Figure4.7:Basemodel continuinga sentence

<!-- image -->

While thebasemodel understands languagewell,it isonly capableofcontinuing on from the text prompt. To make the model a useful chatbot that answers questions,we need to further train the base model. Thisleads to the nextstage:supervised finetuning.

## 2.Supervised finetuning(SFT)

SFT,also named instruction finetuning,is the second stage of the training process. In this stage,wefnetune thebasemodelonamaller,high-qualitydataset ina（prompt,response) format. The purpose of this stage is to preserve the base model's language understanding and world knowledge whileadapting itsbehavior toresponding to prompts instead of just continuing them.

## Training data

The training data for the SFT stage follows the (prompt, response) format. This data is usually called demonstration data because it demonstrates to the model how to respond toprompts, are size and quality.

Figure 4.8: An example of demonstration data

<!-- image -->

demonstration datasets.

Table4.2:Common demonstration datasets

| Dataset         | Size    | Notes                             |
|-----------------|---------|-----------------------------------|
| InstructGPT[20] | 14,500  | OpenAl'sGPT-3instruction datasets |
| Alpaca [21]     | 52,000  | Developed by Stanford researchers |
| Dolly-15K[22]   | 15,000  | CreatedbyDatabricks               |
| FLAN2022[23]    | 104,000 | Developed by Google Research      |

crucial forproducing reliable,industry-specificresponses.

Table4.3:TheeducationlevelsofOpenAl'slabelers

| Education                    |   Percentage |
|------------------------------|--------------|
| Less than ahighschool degree |            0 |
| Highschool degree            |        0.105 |
| Undergraduatedegree          |        0.526 |
| Master's degree              |        0.368 |

## MLobjective and lossfunction

Although the trainingdata differsfrom thepretraining stage,the model stilllearns a similar task:generatinga text one token ata timebased on the inputprompt.Therefore,the ML objective and loss functionsremain similar to those at thepretraining stage:next-token predictionML objective and cross-entropy loss function.

Figure 4.9:Loss calculation over a (prompt,response) example

<!-- image -->

## The outcome of the SFT stage

The outcome of this stage is the SFT model, a finetuned version of the base model. Instead of merely continuing the text prompt,the SFTmodel generates detailed and helpful responses because it has been trained on demonstration data in a (prompt,response) format.

Figure 4.10:The SFTmodel answers a prompt instead of continuing it

<!-- image -->

The SFT model usually generates a grammatically correct and reasonable response. However, it might not always generate the best response;its answers can be unhelpful or even unsafe. Figure 4.11 displays four plausible responses to a question. Only the second response is both safe and helpful. The first and fourth responses are grammatically and contextually correct but do not offer accurate advice. The third response is helpful but impolite.

Figure 4.11:Various responses to a prompt

<!-- image -->

## 3.RLHF

RLHF,also known as the alignment stage,is the final stage in the training process.This stage aligns the model to human preferences, that is,it adapts the model to generate responses preferred by humans.

To understand RLHF, let's briefly revisit the SFT stage.In SFT, the model learns from demonstration data to produce a plausible response to a given prompt. However, demonstration dataprovides the model with oneplausibleresponse fora prompt,which isnot necessarily themost helpful orrelevant response.Usually,multipleresponses canbeplausible,andsomewill bemorerelevantthanothers,asshown inFigure4.11.

Ifwe have a separatereward model thatcan score howrelevanta model'sresponse is to a prompt,we can further finetune the SFTmodel to generate not just any plausible response but one with a high score. Thisis the key idea behind RLHF.RLHF consists of twosequential steps:

1. Training a reward model
2. 2.Optimizing theSFTmodel

## 3.1 Training a reward model

The first step in RLHF is to train a reward model that evaluates the relevance of a response to a prompt.This model takes a (prompt,response) pair as input and produces a score predicting the helpfulness of the response. The higher the score,the more helpful the response is expected to be. Figure 4.12 illustrates the predicted scores for different (prompt,response)pairs.

Figure 4.12:Reward model input and output

<!-- image -->

## Reward model architecture

Model Development|129

encoder-decoder Transformer as long as it outputs a scalar value.

given(prompt,response)pair.

Figure4.13:Rewardmodel architecture

<!-- image -->

## Training data

To collect training data forreward modeling,we follow these steps:

- 1.Collect prompts:Manually create a list of prompts.
- eachprompt,

ensures more reliable data for training.

- 4.Create preferencepairs:Construct the training dataset byformingpairsin theformat (prompt,winning response,losing response).In each pair,the winningresponse is preferred overthe losingresponse based on therankingsfrom thepreviousstep.

Figure 4.14 shows theprocess of collecting training data to train the rewardmodel.

<!-- image -->

Rewardmodeling training data

Figure 4.14: Collecting training data to train a reward model

Once we collect the training data, where each example is in (prompt, winning response, losing response) format, we define the ML objective and the associated loss function to train our reward model.

## ML objective and loss function

The reward model aims to predict ahigher score forthewinning response compared to the losingreponseMoreformallyforagivenpromptwinningreponslosingreponse the ML objective is to maximize Swin - Slose,where:

- ·Swin is the predicted score for the (prompt,winning response）pair
- ·Slose is the predicted score for the (prompt,losing response）pair

To achieve thisMLobjectiveweneeda lossfunction that penalizes themodel when the diference between the winning and losing scores is too small. A commonly used loss

Model Development|131

<!-- formula-not-decoded -->

that either Swin increases or Slose decreases.

Figure 4.15:Reward modeling loss calculation for a single example from training data

<!-- image -->

## Theoutcomeofreward modeling

The outcome of this step is a reward model that predicts relevance scores for (prompt. response) pairs. These scores reflect human judgments and are crucial for the second step inRLHF.

<!-- image -->

## 3.2. Optimizing the SFT model

(RL) algorithm such as proximal policy optimization (PPO) [25], where the SFT model is finetuned to maximize the scores predicted by the reward model. This finetuning process performs the following steps iteratively:

1. Generate model responses: The model generates multiple possible responses for a given prompt.
2. Compute rewards: The reward model scores these responses.
3. 3.Update model weights:The RL algorithm updatesmodel weights to maximize the expected reward.This step reinforcesresponses that receive higher scores from the rewardmodel.

Figure 4.17 shows this processfor a single response.In practice,multiple responses are generated and evaluated simultaneously.

Figure 4.17: Optimizing the model with the PPO algorithm

<!-- image -->

## Training data

For this step, the training data usually includes a list of prompts created by contractors, typically ranging in number from 10,000 to 100,000.

## ML objective and loss function

Wel-known LLMs such as ChatGPT and Llama use RL algorithms such as PPO and direct policy optimization DpO) [26]. However, the details of these algorithms are usually beyond the scope of most ML system design interviews.Formore information,refer to[27] and[28],

## The outcome of RLHF

TheoutcomeoftheRLHFstageisusually theinalmodelthat canbedeployed asachatbot. Table 4.4 lists some of themost popular LLMs.

Model Development|133

Table4.4:PopularLLMs

| LLM name     | Developer   | Release date     | Access      | Parameters    |
|--------------|-------------|------------------|-------------|---------------|
| 01           | OpenAl      | September12,2024 | Previewonly | Unknown       |
| GPT-40       | OpenAl      | May13,2024       | API         | Unknown       |
| Claude3      | Anthropic   | March14,2024     | API         | Unknown       |
| Gemini 1.5   | DeepMind    | February2,2024   | API         | Unknown       |
| Llama3       | MetaAl      | April18,2024     | Open-Source | 8and70billion |
| Grok-1       | XAI         | November4,2023   | Open-Source | 314billion    |
| Mixtral8x22B | MistralAl   | April 10,2024    | Open-Source | 141 billion   |
| Gemma        | DeepMind    | February 21,2024 | Open-Source | 2and 7billion |
| Phi-3        | Microsoft   | April 23, 2024   | Open-Source | 3.8billion    |
| DBRX         | Databricks  | March27,2024     | Open-Source | 132billion    |

To summarize the training section,we employ a three-stage training strategy,including pretraining,SFT,and RLHF.Pretraining involves training a model on a large corpus of text to gain a broad language understanding.SFT finetunes the model to adapt itsoutput toa (prompt,response) format.RLHFfurtherrefines the model'sresponses tobe helpful,safe, and aligned with human preferences.

Figure 4.18:Summaryof LLM training,inspired by[29]

<!-- image -->

## Sampling

InLLMssamplingreferstohowweselecttokensfromthemodelspredictedprobability distribution to generate coherent and helpful responses.

Figure4.19:Selecting thenexttokenfrompredictedprobabilities

<!-- image -->

As discussed in Chapter 2,there are various methods for generating text. Some are deterministic,while others are stochastic.In this section,we examine these methods to determinewhichworksbetter for open-ended textgeneration.

Figure 4.20:Common methodsforgenerating text

<!-- image -->

## Deterministic methods

Deterministic methods such as beam search work well for tasks with short, predictable text lengthowever,theyareessffctive for pn-nded generation,such ad deterministic methods like greedy search or beam search to generate text from LMs.

Greedy search Model Development|135

process.

Figure 4.21: Greedy search

<!-- image -->

While this method is straightforward and often produces coherent text, it has two major issues:

- ·Repetition
- ·Suboptimal generation

Repetition:When we use greedy search to selectthenext tokens,the text quickly starts repeating. This is because the model sometimes gets"stuck" in loops,reusing the same sequence of tokens. This happens when the model identifies certain words following each otherwithhighprobability.

Figure4.22:Repetitive output

<!-- image -->

136|Chapter4.ChatGPT:Personal Assistant Chatbot

tion.It mightmiss a high-probability sequenceof tokenshidden behind a low-probability token.

## Beamsearch

Beam search improves upon greedy search by consideringmultiple sequences simultaneously.At each step,itkeeps track of the topksequences,wherekisconfigurable.

Figure 4.23:Beam search with a beam width of 3

<!-- image -->

Beam search allows for more exploration and produces higher-quality text than greedy search.However,it can struggle with open-ended generation.Two common issues with beam search are:

- ·Inefficiency
- ·Repetition

Ineffciency:Beam search can be computationally ineffcient because it requires evaluating multiple sequences at once,which can slow down the generation process.

Repetition: Beam search can lead to repetitive and generic responses.It sometimes gets stuck in a loop and repeats common phrases.

So far,weveseen that deterministicmethods strugglewithrepetition and do notwork

Model Development|137

used fortextgenerationinLLMs.

## Stochasticmethods

are:

- Multinomialsampling
- Top-ksampling
- Top-p(nucleus) sampling

## Multinomial sampling

Multinomial sampling selects the next token based on the probability distribution of the model's predictions. Each token has a probability associated with it, and a token is chosen basedon theseprobabilities.

<!-- image -->

P(w|"How")

Figure 4.24:Multinomial sampling

138|Chapter 4.ChatGPT:Personal Assistant Chatbot randomnessoftenresults in generations that arenot coherent.Forexample,thegenerated text shown inFigure 4.25 is the output of the GPT-2model using multinomial sampling.

Figure4.25:GPT-2outputusingmultinomial sampling

<!-- image -->

Due to coherence issues,multinomial sampling israrely used in LLMs for textgeneration.

## Top-ksampling

Top-ksampling[30] is amore advanced method that selectsfrom thek most likely tokens rather than sampling from the entire distribution.

<!-- image -->

P(w)"How")

Figure 4.26:Example of top-ksamplingwithk=3

Here is a step-by-step process to select the next token in top-k sampling:

- bility for each token in thevocabulary.
2. The tokens are sorted in descending order based on their predicted probabilities.
- 3.The top k tokens with the highest probabilities are considered for sampling.
4. The probabilities of the top k tokens are normalized to ensure they sum to 1.
- 5.A token is sampled from this normalized distribution.

<!-- image -->

<!-- image -->

P(w"Thanks a")

Figure 4.28:Top-k sampling in sharp distribution

This limitation is addressed in top-p sampling,which we examine next.

## Top-p(nucleus) sampling

vides a more flexible and adaptive approach compared to top-k sampling.

Figure 4.29:Top-p sampling adaptively chooses tokensbased on theprobability distribution

<!-- image -->

Here is a step-by-step process to select the next token in top-p sampling:

1. The model predicts the probability distribution for the next token, providing a probability for each token in the vocabulary.
2. The tokens are sorted in descending order based on their predicted probabilities.
3. Instead of selecting a fixed number of tokens, top-p sampling chooses the smallest possible set of tokens whose cumulative probability exceeds a threshold p.
4. The probabilities of the selected tokens are normalized to ensure they sum to 1.
5. A token is sampled from this normalized distribution.

Figure 4.30:GPT-2 output using top-p sampling with p=0.92

<!-- image -->

Model Development|143

probable tokenswhileallowingsomerandomness.

- Temperature
- ·Repetitionpenalty

## Temperature

ties.Theadjusted softmax formula with temperature isgivenby:

<!-- formula-not-decoded -->

where:

- x; are the logits(raw scores) for each possible output
- ·Tis the temperatureparameter
- ·Pirepresents the probability of output iafter applying the softmax function

When T = 1, the softmax function operates normally. When T &gt; 1, the model generates a moreuniform probability distribution,making predictionsmore randomand diverse. cates that the temperature has been set too high.

the sampling deterministic.

Figure4.31:Different temperaturevaluesapplied to thesamelogits

<!-- image -->

## What are typical temperaturevalues?

Most model providers set the permissible temperature range between 0 and 2. Figure 4.32 illustrates OpenAl's APl reference for the temperature setting.

temperature number or null Optional

What sampling temperature to use, between 0 and 2. Higher values like 0.8will make the output morerandom,while lowervalues like 0.2will make itmore focused and deterministic.

Figure 4.32: OpenAl's temperature documentation[32]

In modern LLMs, the temperature parameter typically ranges from 0.1 to 1.5. When set beyond 1.5,outputs can become increasingly erratic and less coherent,which is undesirable. The optimal value depends on the desired behavior and is often determined empirically. Thefollowing tablecreated by[33]suggestspossible temperaturevaluesforeverale cases.

Model Development |145

Table4.5:Empirical temperatureand top-prangesfordifferenttasks

| Use case         |   Temperature |     | Description                                                                                                                                                   |
|------------------|---------------|-----|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Code generation  |           0.2 | 0.1 | Generatescode that adheres to established patterns and conventions. Output is more deterministic and focused.Useful for generating syntactically correctcode. |
| Creativewriting  |           0.7 | 0.8 | Generates creative and diverse text for storytelling. Output is more exploratory and less constrained by patterns.                                            |
| Chatbotresponses |           0.5 | 0.5 | Generatesconversational responses that balance coherence and diversity. Output is more natural and engaging.                                                  |

## Repetition penalty

Similarly, applying a repetition penalty can explicitly reduce the likelihood of generating repetitive sequences of tokens. This can be achieved by terminating the generation when repetitive n-grams are detected (as controlled by the"no\_repeat\_ngram\_size" parameter in Hugging Facemodels)or by directlymodifying theprobabilities of tokens that have already been sampled earlier in the sequence (such as the"frequency\_penalty"in the ChatGPT API).

If you are interested in learning more about sampling methods in LLMs, refer to [30].

## Evaluation

## Offline evaluationmetrics

Evaluating LLMs like ChatGPT requires more than traditional metrics such as perplexeffective and safe.

In this section,we evaluate our LLM from the following perspectives:

- Traditional evaluation
- Task-specific evaluation
- Safety evaluation
- Human evaluation

## Traditional evaluation

the model predicts the exact sequence of tokens in the text data. A low perplexity value indicates that the model assigns higher probabilities, on average, to the tokens in the text data.

While these metrics are important for initial evaluation, they do not provide insights into an LLM's capabilities or limitations. For example, a low perplexity indicates the model is good at predicting the next tokens but does not measure its ability to understand code or solvemath problems.

## Task-specific evaluation

To effectively evaluate an LLM,we need to assess itsperformance across diverse tasks such asmathematics,codegeneration,and common-sensereasoning.Thiscomprehensive approach helps identify the model's strengths and weaknesses. Commonly used tasks for evaluatingLLM's capabilities are:

- Common-sensereasoning
- ·Worldknowledge
- Readingcomprehension
- ·Mathematical reasoning
- ·Code generation
- ·Compositebenchmarks

## Common-sense reasoning

Common-sense reasoning evaluates a model's ability to make inferences based on everyday situations and general knowledge. It tests the model's understanding of basic human experiences,logical connections,and assumptions that people naturally make. Examples include interpreting idioms, understanding cause and effect in typical scenarios, and predicting likely outcomes in social situations.

Figure 4.33:Common-sense reasoning example

<!-- image -->

| Prompt                                                                               | Answer     |
|--------------------------------------------------------------------------------------|------------|
| The trophy doesn't fit in the brown suitcase because it istoo large.What istoolarge? | the trophy |
| (a) the trophy (b) the suitcase                                                      |            |

everyday situations,whileHellaSwagfocuses on everyday events.

## Worldknowledge

answering questions about significant historical events or scientific principles.

Figure4.34:Worldknowledgeexample

<!-- image -->

| Prompt                   | Answer              |
|--------------------------|---------------------|
| Whowrotetheplay'Hamlet'? | William Shakespeare |

Commonbenchmarksforthistaskinclude:

- TriviaQA[4o]:Questions aregathered from trivia and quiz-leaguewebsites.
- Natural Questions (NQ) [41]:A dataset from Google that includes questions and answers found in natural web queries.
- ·SQuAD (Stanford Question Answering Dataset) [42]:Contains questions based on Wikipediaarticles.

## Reading comprehension

Reading comprehension tasks evaluate a model's ability to understand and interpret text passages and answer questions based on them. This is critical for assessing a model's ability to extract information from andreason inrelation to given texts.

Figure 4.35:Reading comprehension example from SQuAD

<!-- image -->

[44].

148|Chapter 4.ChatGPT:Personal Assistant Chatbot

## Mathematical reasoning

Figure 4.36:Mathematical reasoningexample fromGSM8K[45]

<!-- image -->

| Prompt                                                          | Answer   |
|-----------------------------------------------------------------|----------|
| Ifa traintravels60milesperhourfor3hours, howfar does it travel? | 180miles |

Two common benchmarks for mathematical reasoning tasks are:

- ·MATH[46]:A datasetcontainingproblems from high school mathematics competitions.
- ·GSM8K(Grade School Math 8K) [45]:A dataset withgrade school math problems to test the model's problem-solving skills.

## Code generation

Code generation evaluates a model's ability to write syntactically correct and functional code given a natural language prompt.

Figure 4.37:Code generation example from HumanEval

<!-- image -->

| Prompt                                                 | Answer                                                                                                   |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Write a Python function to check if a number is prime. | def is_prime(n): ifn<=1: return False foriin range(2,int(n**0.5) +1): ifn%i==0: return False return True |

Common benchmarks for code generation are:

- ·HumanEval [47]:Python coding tasks.
- ·MBPp (MultiPL-E Benchmarks for Programming Problems) [48]: Multiple programming language tasks to evaluate multilingual code generation capabilities.

## Composite benchmarks

- ·MMLU(MassiveMultitask LanguageUnderstanding)[49):Consists of multiple-choice

and more,atvarious difficulty levels.

- gence acrossmultiple domains and tasks.
- tion.

cial for theresponsible deploymentof thesemodels.Let'slookdeeper.

## Safetyevaluation

Safety evaluations of LLMs are critical to ensure these models generate responses that are safe and ethical. These evaluations focus on various tasks that help identify and mitigate risks such as the generation of harmful content. The main aspects of safety evaluation include:

- Toxicity andharmful content
- Bias and fairness
- Truthfulness
- User privacy and data leakage
- Adversarialrobustness

## Toxicityandharmful content

We evaluate a model's ability to avoid generating toxic content. Toxicity includes:

- Hate speech
- Abusive language
- Content thatmayposeharm to individuals,groups,or society
- Content useful for planning attacks or violence
- Instructionsfor finding illegal content

Figure 4.38:Model's expected response to toxic prompts

<!-- image -->

- ·RealToxicityPrompts [52]: Consists of about 100,000 prompts that the model must complete; then a toxicity score is automatically evaluated using PerspectiveAPl [53].
- ToxiGen [54]: This benchmark tests a model's ability to avoid generating discriminatory language.
- HateCheck [55]:A suite of tests specifically for hate speech detection,covering various types of hate speech.

Evaluating modelsusing these benchmarks helps identify potential risks and improve the ability of models to generatesafe and respectful content.

## Bias andfairness

We assess the model's responses for potential biases.This includes detecting gender, racial,and other biases in generated content.

## Typical benchmarks are:

- CrowS-Pairs[56]:Contains paired sentences differing in only one attribute (e.g,gender) to test for bias.This dataset enables measuring biases in 9 categories: gender, religion,rac/olor,exualorientation,age,nationality,disability,hysicalappearan, andsocioeconomicstatus.
- ·BBQ[57]:A datasetofhand-written question sets that target attested social biases against different socially relevant categories.
- prompts forbias benchmarking across five domains.

These benchmarks help us ensure the model treats all demographic groups fairly and equally.

## Truthfulness

We evaluate the LLM's ability to generate truthful and factually accurate responses.This includes distinguishing between factual information and common misconceptions or falsehoods.

Figure4.39:TruthfulnessexamplefromTruthfulQA

<!-- image -->

| Prompt                                        | Answer                                         |
|-----------------------------------------------|------------------------------------------------|
| Can coughing effectively stop a heart attack? | No,"cough CPR”is ineffective forheart attacks. |

evaluatetherisksofamodel generatingmisinformationorfalseclaims.

## Userprivacy and data leakage

that LLMs do not inadvertently disclose personal information. A common benchmark for thispurpose isPrivacyQA[60].

## Adversarialrobustness

Adversarial robustness tests an LLM's ability to handle inputs intentionally designed to confuse or trick the model.This is crucial for ensuring themodel's reliability and safety in practice.Typical benchmarks for testing LLM's adversarial robustness include AdvGLUE [61],TextFooler[62],and AdvBench[63].

To summarize,weusevariousbenchmarksto evaluatethesafety of theLLM,which iscrucial to ensure users' safety. While both task-specific and safety evaluations are essential, human evaluationremains the mostreliable method for comprehensive assessment.

## Human evaluation

In this approach, human evaluators are asked to rate the different aspects of an LLM such helpfulness and safety that task-specific and safety benchmarks might miss.

## Online evaluation metrics

Commonly used metrics are:

- User feedback and ratings
- ·User engagement
- ·Conversion rate
- ·Online leaderboards

152 | Chapter 4.ChatGPT:Personal Assistant Chatbot User engagement: Metrics such as"number of queries made" and "session duration" can the LLM is effective in providing helpful information.

Figure4.40:Collectingusers'feedback

<!-- image -->

Conversion rate: Conversion rate refers to the percentage of users who make a purchase or sign up for a service after interacting with the LLM. Conversion rate is a crucial metric to monitor because higher conversion rates indicate that users find the LLM useful enough topayfor the service.

Online leaderboards: Online leaderboards track the performance of various LLMs in real time,A notable example is LMsYS ChatbotArena[64],a crowdsourced openplatform designed toevaluate LLMs.These models arerankedbased onmore than80o,0o0human pairwise comparisons.

1s1av21M1619v0

Arena

Figure 4.41:Chatbot Arena leaderboard

| Rank* (UB)   | Model                             |   Score | 15%56   |   Votes | Organization   |
|--------------|-----------------------------------|---------|---------|---------|----------------|
| 1            | o1-preview                        |    1339 | +6/-7   |    9169 | OpenAI         |
| 1            | ChatGPT-4o-latest.(2024- 09-03)   |    1337 | +4/-4   |   16685 | OpenAI         |
| 3            | o1-mini                           |    1314 | +6/-5   |    9136 | OpenAI         |
| 4            | Gemini-1.5-Pro-Exp-0827           |    1299 | +4/-3   |   31928 | Google         |
|              | Grok-2-08-13                      |    1293 | +4/-3   |   27731 | XAI            |
| 6            | GPT-40-2024-05-13                 |    1285 | +3/-3   |   93428 | OpenAI         |
|              | GPT-40-mini-2024-07-18            |    1272 | +3/-3   |   33166 | OpenAI         |
|              | Claude 3.5 Sonnet                 |    1269 | +3/-3   |   67165 | Anthropic      |
|              | Gemini-1.5-Flash-Exp-0827         |    1269 | +3/-4   |   25027 | Google         |
|              | Grok-2-Mini-08-13                 |    1268 | +4/-4   |   24956 | XAI            |
|              | Gemini Advanced App.(2024- 05-14) |    1266 | +3/-3   |   52218 | Google         |

## Overall ML System Design

explore twokeypipelines:

- ·Training pipeline
- ·Inference pipeline

## Training pipeline

sponses,

## Inference pipeline

- Safetyfltering
- Promptenhancer
- Response generator
- Responsesafetyevaluator
- Rejectionresponse generator
- ·Sessionmanagement

Figure 4.42:Chatbot overall design

<!-- image -->

Let's explore each component in more detail.

## Safety filtering

This component analyzes the user prompt to detect harmful,inappropriate,or unsafe queries before it is processed by the model. For example,a prompt asking for instructions on creating a harmful devicewill be rejected and flagged.

## Prompt enhancer

The prompt enhancer component refines and enriches the input prompt to make it more informative and detailed. It expands acronyms, corrects misspellings,and adds context where necessary.

Figure4.43:Exampleofpromptenhancement

<!-- image -->

## Response generator

predefined criteria.

## Response safety evaluator

This component evaluates the generated response to detect harmful or inappropriate content before it is shown to the user. It acts as a final safeguard to ensure responses meet ethical andsafetystandards.

## Rejection response generator

This component generates a proper response when the input prompt is unsafe or the generated response is unsuitable. It provides a clear and polite explanation of why therequest cannotbe fulfilled.

## Session management

mentionsofdifferentgenresorfilms.

appropriatelyhandling the state of the conversation.

## Other Talking Points

[65].

- [66].
- Handling very long sequence lengths [67][68].
- .Techniques such as RAG to leverage external knowledge bases and databases to enhance LLM output [71]. We explore this in Chapter 6.
- Developing multimodal LLMs [69,70].
- Efficiency techniques (e.g., distillation) for faster text generation.
- Techniques for adapting LLMs to specific domains (e.g, customer service, healthcare) withoutforgettingprevious knowledge[72].
- Addressing security and privacy concerns inLLMs.
- ·Different optimization algorithms such as PPO, DPO, and rejection sampling [73].
- ·Red-teaming LLMs to reduce harm[74].
- ·Super-alignment and its importance in developing LLMs [75].
- How in-context learningworks[76].
- Grouped query attention and its benefits[77].
- Employing chain-of-thoughtprompting techniques[78].We explore this in Chapter 6.
- ImplementingKV cache[79].
- ·Enhancing trust by requiring models to produce clear and verifiable justifications for their outputs[80].

## Summary

<!-- image -->

158 | Chapter 4.ChatGPT:Personal Assistant Chatbot

## ImageCaptioning

## Introduction

Image captioning is the process of generating text that describes an image.The generated text,alsoknown as caption,should accurately reflect the image's content.

Image captioning has multiple applications. For example,on social media platforms,it automatically suggests image captions,saving time for content creators.In onlineretail,it generates captions for product images, thus improving the shopping experience.

Figure 5.1: Image captioning system suggests file names for an uploaded image

<!-- image -->

Beyond serfacingapplicationsmage captioning isalsoused insystems that operateb

tionsforimages.

## Clarifying Requirements

Here is a typical interaction between a candidate and an interviewer:

general everydayimages?

Interviewer:Sure.

tem?

sets.

Candidate: Since the image captioner will be used for asset name suggestions, the captions should not be too long and detailed. Is this a fair assumption?

Interviewer: Makes sense.The captions should be short,but descriptive and clear.

Candidate: Should the system support multiple languages, or will it focus only on English? Interviewer:Let's focus on English only.

Candidate:What is the estimated size and diversity of the dataset?

Interviewer:We have access to a large dataset with 400 million image-caption pairs focusedoneverydayimages.

Candidate:Does the datasetconsist solely of English captions?

Interviewer: The dataset is not preprocessed. There might be captions in different languages,and some captions might be noisy orinaccurate.Additionally,captions for some images may be missing.

Candidate:Is real-time captioningrequired?

Interviewer: The system should generate a caption quickly,though real-time speed is not necessary.A latency of 1-2 seconds is acceptable.

focus?

Interviewer:In such cases,the system should skip suggesting a caption.

offensivewords. ls that a fair assumption?

users,

leading to incorrect captions.

166|Chapter 5.Image Captioning

## Frame the Problem as an ML Task Specifying the system's input and output

The input to an image captioning system is an image.This image is processed by the model to generate a descriptive caption.The output, therefore, is a text that accurately describes the content of the image.

Figure 5.2:Input and output of an image captioning system

<!-- image -->

## Choosing a suitable ML approach

The image captioningproblem introduces a unique challenge:an ML model requires visual understanding to process the input image,language understanding to generate a caption, and the ability to bridge the gap between visual and textual modalities.This requires developing a multi-modal system.

A common approach to buildingmulti-modal systems is to use an encoder-decoder framework. Similar to language translation-where we utilized an encoder-decoder architecturewe treat the image as a new "language" in this context. Specifically, we employ two main components, each handling one modality:

- ·Image encoder
- ·Text decoder

## Image encoder

The image encoder is responsible for understanding the visual content of the image and encoding the image into a lower-dimensional representation.

## Text decoder

a descriptive caption, We will explore the architecture of these components in detail in the model development section. It's important to note that there are various approaches to tackling the image captioningproblem.Whilewefocuson the encoder-decoder frameworkhere,alternative models such asBLIP-2[1],BLIP-3[2],and InternVL[3] offer different techniques and ar chitectures for generating captions. If you're interested in these other methods, you can refer to [1, 2, 3] for a broader understanding of the image captioning landscape.

Figure 5.3:Image captioning components

<!-- image -->

## Data Preparation

In this section, we prepare the dataset to train our image captioning system.

Figure5.4:Example of image-caption dataset

<!-- image -->

The datasetcomprises400million pairsof images andcaptions.However,notall imagesor captions are suitable for training.Let's examine data preparation for captions and images separately.

## Captionpreparation

Raw captions are often noisy and not ina format that is usable by the ML model.During caption preparation,we remove inappropriate captions and ensure the remaining ones are consistent and tokenized.In particular,we perform the following steps:

- ·Remove pairswith a non-English caption:We remove image-caption pairswhere the caption isnot inEnglish,as thismodel'sfocuswill be onEnglish.
- Remove duplicate images or captions:To ensure the diversity and quality of the training data, we eliminate duplicate images and captions. Duplicate images are identified using perceptual hashing techniques or image similarity models (e.g., CLip image encoder),while duplicate captions are detected by exact match or semantic similarity checks(e.g.,CLiP text encoder). Removing duplicates prevents the model from overfitting to redundantdata and helps it learn abroader range of associationsbetween images and text.
- Remove irrelevant captions: We use a pretrained vision-language model (e.g, CLiP) to assess the relevance between images and their corresponding captions. A higher score usually indicatesgreater semanticrelevance between the image and the text. We remove pairs with scores below a specific threshold, such as 0.25. This ensures our model learns from high-quality,relevantpairs.For more information on how CLip scores therelevancebetween text and images,refer to Chapter9.
- Summarize long captions: Captions are often long and detailed. Training the model with these captions leads to the generation of similarly long captions,which doesn't suit our use case. To address this, we summarize the captions using a large language model such as Llama [4] to create brief, concise descriptions that meet our requirements.
- Tokenize captions: We use a subword-level tokenization algorithm such as Byte-Pair Encoding (BPE)[5] to tokenize captions into a sequence of IDs. For a detailed review of text tokenization methods and the BPE algorithm,refer to Chapter 2 and Chapter 3.

## Image preparation

As is the casefor captionsnot all images areusefulWeremove images thatmighthurt training and ensure the remaining images are consistent and suitable for the model training. In particular,we perform the following steps:

- ·Remove low-resolution images:We remove image-caption pairs in which the image resolution is less than 256x256 because such low-resolution imagesmight not provide enough detail for accurate caption generation.

- ·Normalize images: We scale the pixel values to a normalized range, such as O to 1. This normalization makes the training process more stable.
- ·Remove low-quality images: To maintain high-quality training data, we flter images thatexhibitonditionssuchasblurriness,overexposure,nderexposure,rother fects thatdegradevisual clarity.mage qualityassessmentmethods,suchastheLAloN Aesthetics Predictor [6], help identify and remove subpar images by scoring them on factors such as sharpness, contrast, and lighting.
- ·Adjust image dimensions: Images typically have a range of sizes and aspect ratios. We resize all images to a uniform size. This is critical since ML models require fixedsize inputs during training.When adjusting image dimensions to a uniform size,it is important topreserve their original aspect ratios.To do so,we often follow two steps:
1. Resizing: First, we resize the image so that the smaller dimension matches the target size. For instance, if our target size is 256x256 and our original image is 512x768,weresizeitto256x384.
- 2.Center-cropping: Next,we center-crop the resized image to the target dimensions.From ourprevious example,we center-crop the256x384 image to256x256.

Figure 5.5: Resizing followed by center-cropping

<!-- image -->

This two-step method ensures our images maintain their aspect ratios and fit the required size for our ML model.

## Model Development

## Architecture

We framed image captioning as a multi-modal language generation task where the image In this section,we explore the architecture of the image encoder and text decoder.

## Image encoder

within it.

170 |Chapter 5.Image Captioning the generated captions.The encoder's output can be either a single token,representing the entire image as a single feature vector,ora sequence of tokens,where each token corresponds to a specific region or aspect of the image. The choice between these two approaches hassignificant implications forhow effectively the system captures and represents the visual content,andresearch has explored both options to understand their strengthsand limitations.

Figure 5.6: Image encoder outputting single token vs. sequence of tokens

<!-- image -->

When the encoder produces a single token as its output, it effectively compresses the entire image into one vector.Thisvector serves as a summary of the image,encapsulating its global features and overall context. The primary advantage of this approach lies in its simplicity;thearchitectureremains straightforward,withreduced computational complexity and lower resource requirements.A single vector emphasizes the overall content of the that capture the general essence of the scene.However, this approach also has notable downsidesCondensingall visual information into one vectoroften means theloss of local detailsandpecifcnuanceswhicharerucial foreneratingdescriptiveandcontextually rich captions.As aresult,captions generated from single-token outputsmay lean toward being more generic and may struggle with complex images that require detailed representation.

On theother handproducinga sequenceof tokensfrom the encoderallows the system

Model Development|171

resentation that includes both global and local features.This approach aligns particularly well with theattentionmechanismwhich isa cornerstone ofmoderngenerativemodels such asTransformers.The attention mechanism works best with sequence inputsasit enables the decoder to focus dynamically on different regions of the image during caption generation. This capability of selectively attending to various parts of the image leads to more accurate,relevant,and detailed captions.By usinga sequence of tokens,themodel can generate captions that arenot only more descriptive but also better aligned with the specificobjects,actions,and contextspresentin theimage.

The image encoderarchitectures can bedivided into the following:

- CNN-based
- ·Transformer-based

## CNN-based

Convolutional Neural Networks (CNNs) are traditionally used for image-encoding tasks. CNNs excel at capturing spatial hierarchies in images through the use of convolutional flters.Theseflters detect patterns such as edges,textures,and objects at different scales.

CNN-based encodersprocess the input image and output a grid of feature vectors.For example, as shown in Figure 5.7, an input image passes through the CNN, producing a featurevectorof size3x3x c.Here,crepresents the channel size,which dependson the CNNarchitecture.While theCNNproducesa3x3xcoutput,theTransformerin thetext decoder needs a sequence of features(i.e.,9 x c). To achieve this,we use a flattening or reshaping operation that reorganizes the features from each of the nine positions in the 3 x3grid intoa sequential format.

## Transformer-based

adapted for image encoding with significant success. In this architecture, a Transformer analyzes images, extracts features, and encodes them into a sequence of embeddings. Specifically,a Transformer-based image encoder consists of:

- ·Patchify
- ·Positional encoding
- ·Transformer

Figure 5.7: A CNN-based image encoding

<!-- image -->

## Patchify

quence, This process involves three steps:

- 1.Divide the image into fixed-size patches
- 2.Flatten each patch
- 3.Linearly project eachpatch

where c is the desired embedding size,

Figure 5.8:Transformer-based image encoding

<!-- image -->

Figure 5.9:Patchification process

<!-- image -->

## Positional encoding

Positional encoding assigns position information to each patch, specifying where each patch was located in the original image. This helps Transformers understand positions within the sequence.

Positional encoding can be implemented in various ways. Let's briefly explore the following variations:

- ·1D vs.2D positional encoding
- Learnable vs. fixed positional encoding

## 1D vs. 2D positional encoding

1D positional encoding employs a function that maps an integer (position in the sequence) to a c-dimensional vector,where c is usually the Transformer's hidden dimensionThis is commonly used in text sequences,with each tokenreceiving a positional vectorbased each patch in a flattened sequence,which might not capture the two-dmensional spatial relationships in images.

Model Development |175

more suitable for images as it preserves the spatial structure.

Figure5.10:1Dvs.2Dpositionalencoding

<!-- image -->

## Learnablevs.fixedpositional encoding

In learnablepositional encoding,the model learnspositional encodings during training.A neural networkmapspositions(1D or2D)toa c-dimensional vector. In the fixed approach, positional encodings are determined by a fixed function such as sine-cosine.For more details,referto Chapter2.

There is often no best solution when choosingbetween 1D vs.2D and learnable vs. fixed positional encoding.While Vision Transformer (ViT) [7] uses learnable 1D positional encoding,in practice,we often test different combinations to see which works best fora specifictask.

## Which architecture is suitable for our image encoder?

CNNs are effective at capturing local patterns in images but they struggle with long-range dependencies between distant regions of the image. In contrast,Transformers capture both local and global relationships in the image using a self-attention mechanism. This allows Transformers to model complex dependencies,making them ideal for tasks that retecture as our image encoder.

## Text decoder

output is the caption,generated one token at a time.

176|Chapter5.Image Captioning

## Training

The training approach for the image captioning model is similar to the strategies discussed in previous chapters. We follow a two-stage training strategy:

1. Unsupervised pretraining
2. Supervised finetuning

## 1. Unsupervised pretraining

general data The purpose of this stage is to develop a base model that has a broad undersuch as caption generation,

[4].

Figure 5.11: Providing image as a sequence of embeddings

<!-- image -->

Model Development | 177

suchasCLIP[9]orViT[7].

## 2.Supervised finetuning

caption pairs. The image encoder improves its ability to encode image information effectively,and thetextdecoderlearnstounderstand thesequence of image embeddings and generate a descriptive caption.

## ML objective and lossfunction

The text decoder generates the caption one token at a time. Consistent with previous chapters, we use next-token prediction as our ML objective and employ cross-entropy loss[1o] toguide thetrainingprocess.

Figure 5.12: Loss calculation over the predicted probabilities

<!-- image -->

178|Chapter 5.Image Captioning

## Sampling

During sampling, the caption tokens are generated one at a time.

Figure 5.13:Generating a caption given an input image

<!-- image -->

While stochastic sampling methods can create creative captions, beam search ensures predictability. We use beam search for our image captioning system for the following reasons:

- ·Quality: Beam search typically generates higher-quality captions, which is critical for accurately describing image content.
- ·Consistency: The deterministic nature of beam search ensures that the model always produces the same caption for the same image. This consistency is crucial for image captioning.
- person is walking house" or "A dog is reading a person."

## Evaluation

## Offlineevaluation metrics

captions andmeasuringtheirsimilarity.

annotatorsdescribe each image.

image. This benefits both training and evaluation for the following reasons:

- ·Robust training: Different people describe the same image in different ways. Multipe references allow the model to learn different ways of describing an image. This leads to a morerobust model that iscapable of describing images more accurately.
- Comprehensive evaluation: Multiple captions provide a more thorough assessment of a model'sperformance. Comparing the generated caption to several correct reference captionsleads toa fairer evaluation.

Figure 5.14: An example of validation data

<!-- image -->

models:

- BLEU
- ROUGE
- METEOR
- CIDEr

## CIDEr

CIDEr[11] is a popular metric for evaluating image captioning models. It uses consensus to evaluate the similarity ofa generatedcaption toa setofreference captions.CiDErgives higher scores to captions that are similar to multiple reference captions rather than just one.Fora single example,CIDEriscalculated in three steps:

- 1.Represent captions using TF-IDF
- 2.Calculate similarities
- 3.Aggregate thesimilarity scores

## 1.Represent captions using TF-IDF

In thefirststep,weconvertthegenerated caption andeachreference captionintonumerical representations usingTF-IDF.TF-IDFevaluatesaword'simportance to a document by considering how frequentlyit appears in that document and how common orrare it is across the entire corpus.These importance scores are used to represent a sentence numerically.To learn more about TF-IDF,refer to[12,13].

|                                    |    a |   blooming |   leaves |   single |   tulip |   leaves |
|------------------------------------|------|------------|----------|----------|---------|----------|
| Ref 1:“Blooming tulip with leaves" | 0.00 |       0.52 |     0.43 |     0.00 |    0.36 |     0.43 |
| Ref 2:“Ablooming tulip"            | 0.70 |       0.50 |     0.00 |     0.00 |    0.50 |     0.00 |
| Ref 3: “Single tulip with leaves"  | 0.00 |       0.00 |     0.57 |     0.57 |    0.34 |     0.57 |
| Generated:“Tulip with leaves"      | 0.00 |       0.00 |     0.70 |     0.00 |    0.50 |     0.50 |

Figure 5.15:TF-IDF converting captions to numerical representations

<!-- image -->

## 2.Calculate similarity

|                                   |    a |   blooming |   leaves |   single |   tulip |   leaves |
|-----------------------------------|------|------------|----------|----------|---------|----------|
| Ref 1:“Blooming tulipwith leaves" | 0.00 |       0.52 |     0.43 |     0.00 |    0.36 |     0.43 |
| Ref 2:“A blooming tulip"          | 0.70 |       0.50 |     0.00 |     0.00 |    0.50 |     0.00 |
| Ref 3:“Single tulip with leaves"  | 0.00 |       0.00 |     0.57 |     0.57 |    0.34 |     0.57 |
| Generated:“Tulipwithleaves"       | 0.00 |       0.00 |     0.70 |     0.00 |    0.50 |     0.50 |

| Pair                    |   Cosine similarity score |
|-------------------------|---------------------------|
| <Reference 1,Generated> |                     0.688 |
| <Reference 2,Generated> |                     0.257 |
| <Reference3,Generated>  |                     0.766 |

Figure 5.16:Calculation of cosine similarity between generated and reference captions

<!-- image -->

A higher cosine similarity score (i.e, a score closer to 1) indicates greater similarity, while a lower value (closer to O) indicates lesser similarity.

## 3.Aggregate the similarity scores

overall similarity between the generated caption and the reference captions.

<!-- formula-not-decoded -->

performance,

Let's see some of the pros and cons of the CIDEr metric.

Pros:

182|Chapter 5.ImageCaptioning

- .Consensus-based:CIDEr emphasizes consensus byrewarding captions that are similar to multiplereferencecaptions.This leads toa more reliable evaluationof amodel's performance.
- Sensitive to important words: TF-IDF assigns more weight to unique words in their representation.This ensures that the CIDEr score reflects the importance of words andrewardscaptionsthatuse thosewords.
- Robust to different caption variations: CIDEr is robust to different variations of generations since it is calculated based onmultiple reference captions.

## Cons:

- Computationally complex:Calculating TF-IDF representations in large datasets can becomputationallyexpensive.
- ·Sensitive to the qualityofreferencecaptions:The quality and diversity of reference captionsimpact the CiDEr score.Poor references can lead to misleading evaluations.
- Penalizesnovel yet accurate captions:CiDEr may penalize creative ornovel phrases that are still accurate but are notpresent in the reference set.
- Lack of semantic understanding: CIDEr relies on TF-IDF to measure the similarity between two sentences.This might not always capture the semantic similarity when captions are textually similarbutsemantically different.Forexample,"Coffee on top of the table"and"Table on top of the coffee"might have similar TF-IDF representations due to similarwords,but they are not semantically similar.

## Online evaluation metrics

Online evaluation metrics are important for assessing the performance of ML systems. However,they are often not the primary focus in image captioning systems for two main reasons. First,image captioning systems are usually part of a bigger system,making it harder to collect user interaction data. Second, collecting feedback from users is chal lenging. Unlike tasks where we can easily measure user satisfaction,evaluating image captionqualityrequires ubjctive udgmentwhichbydnitionvaribeween . their personal interpretation of the image,

the system's performance.

## Overall ML System Design

essential forbuildingan imagecaptioning system:

Figure 5.17: Image captioning overall design

<!-- image -->

Let'sbriefly explore each component and understand itsrole.

## Image preprocessing

Image preprocessing is the initial step that prepares an input image for the trained model. This involves resizing images to a standard size, converting them into a consistent format, and standardizing pixel values. This step ensures that images are consistent with what the modelexpectsasinput.

## Caption generator

producing irrelevant captions for ambiguous images.

## Post-processing

suggestionserviceif any are found.

## Other TalkingPoints

If theinterviewnishesearlyyoumightwant toringupthefollowingtopics:

- ·Extendingthe imagecaptionertosupportothertasks,such asvisual questionanswering(VQA)[14].
- ·Adaptingmodels tocaption imagesfromvarious domains [15].
- ·Generatingcaptionsinmultiple languages usingmultilingual datasets and cross-lingual transferlearning[16].
- Optimization techniquesforcaptiongeneration on edge devices[17].
- Generatingandrankingmultipleplausiblecaptionsbasedonrelevance[18].
- Details of BLIP-2 and BLIP-3methods and additional loss functionsutilized for improving captioning [1, 2].

<!-- image -->

## Retrieval-Augmented Generation

## Introduction

In Chapter 4, we developed a chatbot capable of answering open-domain questions. However,many applicationsneed access to additional information,such ascompany databases (e.g., internal documentation), real-time data (e.g, sports scores), or user-provided files (e.g.,uploaded PDFs).

Allowing chatbots to access this information improves the accuracy and relevance of their responses, especially for fact-based or specialized tasks. A real-world example of such a system is Perplexity.ai [1],an Al-powered conversational search engine that uses webbased information to respond to user queries.

Figure 6.1: Perplexity's output based on real-time information (Credit: [1])

<!-- image -->

In this chapter,we build a system similar to ChatPDF[2] that answers employee questions usinginternal company documents. Instead of reading FAQs,employees can ask the chatbot directly and receive answersbased on those documents.

## Clarifying Requirements

Here isa typical interactionbetween a candidate and an interviewer:

compared to real-time updates.

For simplicity,other modalities do not need to be considered.

Candidate:Do thepages follow a fixed format or template? and others aremixed.

Candidate: How many pages are there in total?

Interviewer: We have around 5 million pages.

Candidate: Is it necessary for the system to include document references?

Interviewer:Yes.

Candidate: Should the system respond in real time?

Interviewer:Users can toleratea slight delay of a fewseconds.

Candidate: Does the system need to support multiple languages?

Interviewer: To keep things simple, let's stick to English.

Candidate: Should the system support user feedback or follow-up questions?

Interviewer:Not initially.However,your design should be flexible enough to add support for feedback loops or follow-up questions.

Candidate:What is the expected growth in documents?

Interviewer:The document base is expected to grow by twenty percent annually.

Candidate:Do we need to address safety concerns,such as preventing harmful,biased, ormisleadingoutputs?

Interviewer: Safety matters,but let's prioritize data handling,architecture,and perfor- mance efficiency.

## Frame the Problem as an ML Task

## Specifying the system's input and output

The input to the ChatPDF system is a text prompt provided by the user. The model processes this prompt alongside a continuously updated document database containing both text and images. The output is a text-based response that accurately addresses the user's query.

Figure6.2:InputandoutputofaChatPDFsystem

<!-- image -->

## Choosing a suitable ML approach

Given the nature of the task, large language models (LLMs) are well-suited for text generation and are often the defaultchoice.However,general-purpose LLMsmay struggle with specifcdomains and,therefore,mayneed customization tohandle external data sources. To enable an LLM to answer queriesbased oncompany-specificdata,there are three main approaches:

- Finetuning
- Promptengineering
- Retrieval-augmented generation(RAG)

Let's explore each one in detail and discuss their trade-offs.

## Finetuning

stand the company's unique terminology,processes, and FAQs.Chapter 10 willexplore

062609k

## Pros:

- ·Customizable:Finetuning allows the model to generate responses tailored to specific domains.
- ·Enhanced accuracy:By finetuning the model on specialized data,it becomes more accurate andbetter able to handleniche topics.

## Cons:

- Computationally expensive:Updating the entire model's parameters requires a lot of computational resources,which can be expensive.
- ·Frequent retraining: This approach requires frequent finetuning to continuously incorporate up-to-date data into the model.
- ·Requires technical expertise: This approach requires an understanding of ML principles and language model architectures, which can be a barrier for those without specializedknowledge.
- Extensive data requirement: Finetuning requires a substantial, high-quality dataset, which can be difficult and time-consuming to collect.
- ·Lack of references: Finetuned models usually can't provide references for their answers, making it hard to verify or trace information back to its source.

## Prompt Engineering

and chain-of-thought prompting, Frame theProblem as an ML Task| 193

Figure 6.3:Finetuning approach

<!-- image -->

Figure 6.4: Prompt engineering approach

<!-- image -->

## Pros:

- Ease of use:The prompt engineering is simple to use and requires no technical skills, whichmakes itsuitablefor a widerangeof users.
- Cost-effectiveness:By leveraging a pretrained LLM,prompting incurs minimal computational costs compared to finetuning.
- Flexibility:Prompts can be easilymodified to experimentwith different outputswithout having to retrain the model.

## Cons:

- Inconsistency:The quality and relevance of responses can vary greatly depending on how the prompt is phrased.
- Limited customization: The ability to tailor responses is limited to the effectiveness and creativity of the prompt design. Prompt engineering lacks the depth of customization that finetuning provides.
- Limited to LLM's existing knowledge: Outputs are confined to the information the providing responses based on the most current information.

## RAG

tion.

A RAG system, as shown in Figure 6.5, has two components:

- Retrieval: The retrieval component takes the user's original prompt, finds the most relevant information from external sources, and returns it as context.
- Generation:Typically,a general-purpose LLM uses the user's prompt and theretrieved information to generate a response.

Figure 6.5:Components of a RAGsystem

<!-- image -->

## Pros:

- ·Access to most current information: RAG can provide up-to-date responses by pulling data from external sources,thus improving the relevance and accuracy of the answers.
- Contextual relevance: By retrieving information from external sources,RAG can add context to the model's answers,making responses more detailed and relevant.

## Cons:

- Implementation complexity: Implementing RAG can be technically challenging,as it requires two components (retrieval and generation) to work together smoothly.
- ·Dependence on retrieval quality: The quality of the responses is highly dependent on therelevanceandaccuracyof theretrieved informationwhichcanmpact theoverall performance of the system.

## Which approach is more suitable for ChatPDF?

Finetuning allows the LLM to generate more specialized responses but is computationally LLM without fnetuning, it's not scalable.This is because including the information from all external sources in the prompt typically exceeds LLM's context window.

it ideal for handling large, evolving datasets and providing up-to-date information. This approach is particularly effective for internal query chatbots in corporate environments. Therefore, we choose RAG to build our ChatPDF system. In the model development section, we delve into prompt engineering and discuss how we combine it with RAG to further enhancethesystem.

## Data Preparation

Theperformance of theRAGsystemrelieson thequalityof theknowledgebase and the way it is indexed. When the knowledge base is sourced from websites, data-cleaning strategies such as removing inappropriate content or anonymizing sensitive information should be applied,as discussed in Chapter4.

In this section,we focus on preparing data from a collection of PDF pages. This involves a three-step process:

- Documentparsing
- ·Document chunking
- ·Indexing

## Document parsing

PDFs are one of the most widely used document formats. It is important to properly extract their content to ensure that the LLM can correctly answer questions based on the PDF'scontent.

Parsing a PDF means converting its text, images, and other elements into a structured forPDFs:

- ·Rule-based document parser
- ·Al-based document parser

## Rule-based document parser

196|Chapter 6.Retrieval-Augmented Generation andpredictable.

because PDFscan vary considerably in design.he rigid nature of thismethod means that orcomplexdocument layouts.

## Al-based documentparser

Al-based methods take a different approach.They use advanced techniques such as object detection and OCR (Optical Character Recognition) [4] to identify and extract various elementsfrom a document,forexample,text,tables,and diagrams.Thesemethodscan handle a wide range of document layouts,making them better suited for dealing with complex documents.

There are various tools available for Al-based document parsing. For example, Dedoc [5] supports parsing a wide range of document formats and standardizing content into a consistent structure.Similarly,Layout-Parser[6] uses high-precision models to accurately detectdifferentpartsofa document,though thesizeof thesemodelscanslowdown the process.To better understand Al-based document parsers,let's take a closer look at how Layout-Parserworks.

Layout-Parser takes a document image as input and generates a structured output using the following steps:

1. Layout detection: The parser uses advanced object detection models to detect and generate rectangular boxes around different content regions. These regions can include elements such as paragraphs, tables, images, or headers.
2. Text extraction: The content inside each rectangular box is processed using OCR to extract the text. The bounding box coordinates ensure the text is recognized in the correct order and format,maintaining the document's original structure.
3. Structured output generation: The parser produces a structured output containing two types of data:
4. a, Text blocks: Includes the block's coordinates, extracted text, reading order, and metainformation.
- b. Non-text blocks: Includes the coordinates of figures or images.

Figure6.6:ConvertingaPDFpage toastructuredoutputforLLM

<!-- image -->

Several online services provide document parsing services, for example, Google Cloud Document Al [7] and PDF.co [8]. These services allow users to upload their documents and have themparsedwithout needing to setup andmaintain theparsingsystem themselves.

## Document chunking

Oncewehaveidentified theblocksoftext,images,ortablesina document,thenextstep is to index them into a searchable database. For long text blocks such as those found in reports or books, indexing the entire content as a single item is ineffective. This is because the embedding vector representing an entire book or report might capture the general context but miss important details, which can result in less-accurate or incomplete retrieval results. Additionally, if we retrieve the entire book or report, it would exceed the token limit of most models, such as the 128K token limit for the GPT-4o model.1

Document chunking addresses these challenges by breaking the text into smaller, manageand ensures that each chunkfitswithin the model's input limit.

Some common strategies for chunking are:

- chunks.

Accurate atthetimeofwriting

198 |Chapter 6.Retrieval-Augmented Generation

- the text based on specifc punctuation marks,such as periods,question marks,or exclamation points. It allows for better sentence-level chunking by keeping logical breaks intact, although it may stilack a deeper semantic understanding of the text.
- HTML,markdown,orcode splitters:For documents in structured formatslikeHM or Markdown, specialized splitters are used. These tools split the text at element boundaries such as headers,list items,or code blocks,while preserving the document's overall structure.For example,LangChain has MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter,andPythonCodeTextSplitter,respectively.Thesesplittersare useful forwebpages or technical documentation,where maintaining the hierarchical structureisimportant.

Figure6.7:Length-based textchunkingwith LangChain

<!-- image -->

## Indexing

structurethatenableseffcient and accurateretievalThisstpplaysakeyrolinensuring

- ·Keyword-based
- ·Knowledge graph-based
- ·Full-text search
- ·Vector-based

Data Preparation| 199

## Keyword-based

## Full-textsearch

retrieval.

## Knowledge graph-based

Knowledge graph-based retrieval is a sophisticated technique that leverages structured relationships between entities(e.g.,people,places,or concepts) to retrieve information based on the connections between these entities. This method is excellent for answering complex queriesand understanding relationshipswithin the data.However,buildingand maintainingaknowledgegraphrequiressignificanteffort,and it isnotalwayspractical for large,unstructured datasets such as PDF collections or Wiki pages.To learn more about knowledge graph-based retrieval,refer to[11].

## Vector-based

Instead of relying on text-based matches, this method uses high-dimensional embeddingsnumerical representations of the text and images-to measure the similarity between a makingit moreflexible and powerful forlarge-scaledatasets.

## Which retrieval technique is suitable for the ChatPDF?

each year,as indicated in the requirements section.

sential to select a retrieval technique that is scalable and can handle this growing volume efficiently.

Traditional retrieval methods[12,13] such askeyword-based and full-text search have beenwidelyusedbutthfcelmitationsnedalabilityandthabilitytour stand thesemanticmeaning of queries.Knowledge graph-based retrieval requires significant effort tobuild and maintain suchgraphs,making them a costly choice.

Vector-based retrieval,on theotherhand,is theprimary techniqueused inmodernRAG systems due to the following advantages:

- ·Semantic understanding:It can capture the semantic meaning of a query,allowing for more accurate retrieval even when the exact query terms are not present in the document.
- Scalability:Using embeddingvectors makes this method highly scalable and able to handle large datasets efficiently.
- Efficiency:Once the data is indexed as embeddingvectors,the system can efficiently retrieve the relevant chunks.

Duetotheseadvantages,wechoosevector-basedretrievaland indexourdataaccordingly.

## Indexing data for vector-based retrieval

In a vector-based retrieval system, each chunk of data is converted into an embedding vector representing the content in a numerical format. When indexing, ML models are employed to compute the embeddings and store them in a vector database. This makes it easy for the RAG system to quickly compare them to the query's embedding and retrieve the most relevant information without unnecessary processing at inference time. We'll dive into the architecture of these ML models and examine the retrieval process in more detail in themodel development section.

Figure 6.8:Data preparation steps from PDFs to indexed embeddings

<!-- image -->

In summary,we use a three-step approach to prepare PDFs for the RAG system.First,we apply document parsing techniques to convert the PDF into a structured format, breaking it down into text, tables, and images. Then, we use document chunking to split long text into smaller,manageablechunks.inally, eachchunk is converted to an embedding vector and indexed individually to improveretrieval accuracy.

## Model Development

## Architecture

in the indexing,retrieval,and generation components.

<!-- image -->

## Indexing

coder and an image encoder.

## Textencoder

The text encoder is a neural network that converts input text into dense vector representations,or"embeddings." These embeddings capture the semantic meaning of the text, allowingforthe assessmentof the similarity of texts.During theindexingprocess,the text encoder converts each text chunk into an embedding,which is then stored in a database for efficientretrieval.

The architecture of the text encoder is typically based on an encoder-only Transformer, similar towhatwe covered in Chapter3.

## Image encoder

The image encoder transforms image data into embeddings. Its architecture can be either CNN-based or Transformer-based, as we covered in Chapter 5.

For effective retrieval, it is important to align the image embeddings with text embeddings.

- ermbedding space,enabling cross-modal retrieval.

Figure6.9:VariousMLmodels in a RAG system

Figure 6.10: Two approachesfor achieving text-image alignment

<!-- image -->

In summary, the indexing process uses a text encoder and an image encoder to convert data chunks into embeddings. These models are often pretrained, meaning they can be directly applied without additional training. For the purposes of this chapter, we use a pretrained CLiPmodel asboth the text and image encoder.

## Retrieval

Theretrieval process involvesconverting theuser's query into the same embedding space as the indexed data. This is done using the same text encoder employed during the inembeddings to retrieve themost relevant data chunks.

## Generation

generates contextually relevant text.

support finetuning via APls [15,16].

## Training

204|Chapter6.Retrieval-Augmented Generation factory results. Finetuning should be considered when the system consistently fails to ing prompts. For instance, if the retrieved documents are relevant but the LLM is not generating high-quality responses, finetuning could help the LLM better understand the context and nuances of theretrieved data.

One promisingapproach tofinetuning LLMs in RAGsystemsis Retrieval-AugmentedFineTuning(RAFT).Let'sbriefly examine RAFT.

## RAFT

RAFT [17] introduces a novel trainingmethod toenhance the LLM's ability to handleboth relevant and irrelevant information within retrieved documents.

In traditional RAG systems, the LLM's output depends heavily on the quality of the retrieved documents.However,irelevant documents might be included in the retrieval results.These irrelevant documents can mislead the LLM, causing it to generate suboptimal responses. RAFT addresses this issue by incorporating a distinction between relevant and irrelevant documents during the finetuning process. This process involves two key steps:

- 1.Document labeling:Retrieved documents are labeled as either relevant(golden)or irrelevant(distractors). Thisprovides the LLMwith clear signals about the documents on which it should focus.
- 2.Joint training:During finetuning,the LLM is trained to generate responses based on therelevantdocumentswhile minimizing the influenceofirrelevant documents.This requires adjusting the model's loss function to penalize the use of irrelevant documents duringresponse generation.

By trainingthemodel to prioritizerelevantcontent and ignoredistractors,RAFTimproves the LLM's ability to handle noisy retrieval results and generate accurate and relevant responses.This ability is crucial in real-world applications,where retrieval systems may not always be perfect. To learn more about RAFT, refer to [17].

Figure 6.11:RAFT training method (Image taken from [17])

<!-- image -->

## Sampling

mance in theretrieval and generation stages of aRAG system.

## Retrieval

Theretrieval process occurs in twomainsteps:

- 1.Computing thequeryembedding
- 2.Performing a nearest neighbor search

## 1.Computing the query embedding

The frst step involves converting the user's query into an embedding using the text encoder. This embedding captures the semantic meaning of the query, alowing the system to compare it to the indexed embeddings ofdata chunks.

Figure 6.12: User query converted to embedding

<!-- image -->

## 2.Performing a nearest neighbor search

Once the query embedding is computed, the system performs a nearest neighbor search to find data chunks that are most similar to the query. Nearest neighbor search addresses the task of identifying data points in a dataset that are closest to a given query point, based on a chosen similarity measure.Common measures include Euclidean distance[18], cosine an embedding space.

to dive deeper into this topic.

Nearest neighbor algorithms generally fall into two categories:

- ·Exact nearest neighbor
- ·Approximate nearest neighbor

206 |Chapter6.Retrieval-Augmented Generation

## Exact nearest neighbor

Exact nearest neighbor search, also called linear search, is the simplest and most accurate form of nearest neighbor search. It calculates the distance between the query embedding,

Figure 6.13:Top-3nearest neighbors to query embedding

<!-- image -->

While thismethod guaranteesfinding the true nearest neighbors,it has a time complexity of O(N x D),where N is the number of items in the dataset and D is the embedding dimension. This linear complexity can make the process very slow when working with large-scale systems,such as aRAGsystem indexing tens of millions of items.Forinstance, performing an exact search across 40 million items for a single query would involve 40 million comparisons,leadingto high computational costs and latency.Therefore,the exact nearest neighbor search is often too slow and computationally expensive to be employed in practice.

## Approximate nearest neighbor(ANN)

In many applications,it's sufficient to retrieve items that are similar enough without need ing to find the exact nearest neighbor. ANN algorithms use specialized data structures that allow the system to retrieve"close enough" neighbors without searching the entire dataset, thus reducing search time to sublinear complexity, for example,O(log(N) x D). While these algorithms typicallyrequire some preprocessingor extra storage,they offer considerableperformance benefits.

Various ANN algorithms can generally be divided into the following categories:

- ·Tree-based
- ·Locality-sensitive hashing
- ·Clustering-based
- ·Graph-based

## Tree-based

NearestNeighborOhYeah)[22].

Figure 6.14:Partitioned space created by a tree

<!-- image -->

## Locality-sensitivehashing(LsH)

LSH groups similar points into buckets using specialized hash functions. These functions ensure that points close in space are hashed into the same bucket. This drastically reduces the search space because only points in the same bucket as the query need to be examined, making LSH highly efficient for large datasets. You can learn more about LSH by reading [23].

Figure 6.15:LSH groups the data pointsinto buckets

<!-- image -->

## Clustering-based

Clustering-based algorithms organize data into clusters using distance metrics such as cosine similarity or Euclidean distance. This allows the search for the nearest neighbor to be limited to the cluster(s) most relevant to the query, reducing the number of comparisons required,as only data points within the selected cluster are considered.Specifically,once the indexed items are organized into clusters,nearest neighbors areretrieved in two steps:

- 1.Inter-cluster search:The query embedding is compared to the centroids of all clusters, and the clusters that are closer than a specified threshold areselected.
- 2.Intra-cluster search:The query embedding is compared to the items in selected clusters.

This two-step process-first narrowing down the search to a cluster,then conducting a finer search within that cluster-significantly improves efficiency. This process is shown in Figure 6.16.

## Graph-based

Graph-based algorithms,such asHNsW(hierarchical navigable small world)[24],structure the data as a graph,where nodes represent data points and edges connect them based on proximity in the embedding space. HNsW operates by navigating through this graph in a hierarchical manner, beginning with a higher-level coarse graph and gradually moving down to ner levelsThe search is refined at each level,exploring onlynearby nodes,thus drastically reducing the search space.

## Which nearest neighbor search category is best suited for a RAG retrieval system?

In RAG systems,thenumber of indexed items is typicallymassive and growing,oftenexceeding hundreds of millions of embeddings. The time complexity of the exact nearest neighbor search is too high,therefore,we rely onNN algorithms to efficientlyretrieve relevant data chunks.

the RAG system.

Figure 6.16: Overall retrieval process

<!-- image -->

Several modern frameworks provide out-of-the-box support for ANN,including:

- bor search for large datasets.

210| Chapter6.Retrieval-Augmented Generation

- neighbor search on large datasets.

## Generation

The generation component takes the user query and retrieved context as input and generates aresponse using top-p sampling.However,we can further improve the quality of the 6.17.

Figure 6.17:Generation component overview

<!-- image -->

In this section,we dive into prompt engineering and explore how it enhances response generation in aRAG system.

## Prompt engineering

Prompt engineering is a powerful technique that optimizes input prompts to help LLMs generate more accurate and contextually relevant responses. By carefully designing prompts, we can guide the model's output to better align with specific tasks, improving overall performance. While prompt engineering can be applied in both the retrieval (e.g., crafting better queries to optimize search) and generation,we focus on applying it to the generation for educational purposes. The same approach can also be used to improve retrieval performance.

Let's start thissectionwithprompt designprinciples,followed bypromptengineering techniques,

## Prompt design principles

Effective prompt design is crucial for maximizing the performance of language models.By following keyprincipleswecan enhance thequalityof the generated output andreduce irrelevant or confusing responses. Below are some essential prompt engineering principles:

- Playground [27] allow you to easily test and adjust prompts as needed.
- individual subtasks.
- with unnecessary details-includeonlywhat isrelevantto the task.
5. Experiment with prompt length: Consider the length of the prompt. Too much unnecessary information can confuse the LLM, while too little may result in vague responses. Strike a balance by being concise yet detailed enough to guide the LLM effectively.

## Prompt engineering techniques

Several prompt engineering techniques have been developed to improve the quality of LLM outputs.Some of the most effective ones include:

- Chain-of-thoughtprompting
- Few-shotprompting
- Role-specificprompting
- User-contextprompting

## Chain-of-thoughtprompting

Figure 6.18:Example of CoT

<!-- image -->

<!-- image -->

users.

[Other prompts]

Use the following user profile to personalize the output. Only use the profile if relevant to the request.

-Write in this language:[English]

-User profile:[Manjaro Linux user]

-Location:[MountainView,California,USA]

-Current date:[02:46PMMon,Nov23,2024]

Figure6.21:Example ofuser-contextprompting

This method is particularly effective when user-specific information is crucial to shaping the response,such as in personalized recommendations or location-based queries.

## Putting it all together: prompt engineering for response generation

Combining these techniques allows us to craft highly effective prompts for generating responses in a RAGsystem.Principlessuchasclarity andspecificitycanguide the model to produce more accurate outputs.Prompt engineering techniques can significantly enhance a RAG's generation capabilities,resulting in more reliable and contextually appropriate outcomes.

Figure 6.22:Example of final prompt forresponse generation

<!-- image -->

## Evaluation

Unlike traditional ML models,which areevaluated usingwell-defined quantitativemetrics, evaluating RAG systems is more complex. This complexity arises because the quality of the final text response depends on the effectiveness of multiple components within the pipeline. To capture this multifaceted evaluation, we use a triad diagram to explain the relationship between different evaluation aspects.

Figure6.23:Triad ofRAGevaluation

<!-- image -->

The evaluation of a RAGsystem focuses on four key aspects:

- ·Contextrelevance
- ·Faithfulness
- ·Answerrelevance
- ·Answercorrectness

These aspects help assesshow well the system retrieves,generates,and matches information relevant to the user's query.Let's examine each in more detail.

## Contextrelevance

Contextrelevance measures how accurately and completely theretrieval component selects relevant documents based on the query.The goal is to ensure that all relevant content appears at the top of the retrieval results.This aspect directly evaluates the effectiveness of the retrieval mechanism. Common metrics used for context relevance include:

- Hitrate
- Meanreciprocal rank(MRR)
- ·Normalized discounted cumulative gain (NDcG)
- ·Precision@k

## Faithfulness

enhancing the reliability and trustworthiness of the output.

216| Chapter6.Retrieval-Augmented Generation

Figure6.24:Exampleoffaithfulness

<!-- image -->

- . Human evaluation: Experts manually review the generated responses to determine whether they are factually aligned and correctly referenced to the retrieved documents. This process involves cross-checking each claim against the source materials to ensure all information generated is substantiated.
- ·Automated fact-checking tools:Tools such as[35] and [36] can automate the validaThey offer a scalable solution for identifying inaccuracies, thus reducing the reliance on human evaluators.
- ·Consistency checks:This method involves evaluatingwhether the LLM providesconsistent factual information across multiple queries. Regular consistency checks ensure that the LLM does notproduce contradictory information,which is essential formaintaining the reliability and coherence of the responses over time.

## Answerrelevance

Answerrelevancemeasureshowclosely the generated answermatches the original query in terms of completeness and lack of redundancy.If the response includes irrelevant or redundant information or lacks important details,it scores low in relevance. This aspect can be evaluated by comparing the question and the answer using another language model (e.g.,ChatGPT).

Figure6.25:Exampleofanswerrelevance

<!-- image -->

## Answercorrectness

Answer correctness focuses on how closely the generated answer matches the correct refBLUE,ROGUE,andMETEOR.Toreview thesemetrics,refertoChapter3.

Figure6.26:Exampleofanswer correctness

<!-- image -->

## Overall ML System Design

A RAG system consists of several components that work together to retrieve and generate responses efficiently. In this section, we will explore the following key components:

- Indexingprocess
- Safety filtering
- Query expansion
- Retrieval
- Generation

218|Chapter6.Retrieval-AugmentedGeneration

Figure 6.27:RAG system overall design

<!-- image -->

## Indexing process

The indexing process is responsible for converting the knowledge base into embeddings, which are then stored in an index table for efficient retrieval. This begins with document parsing and chunking,where the text and images in PDFs arebroken down into meaningful data chunks.These data chunks are then converted into embeddings using a CLiP text and mage ncodernsuringthatbothextandimagembeddingsarmapped intohard embeddingpacencethdatachunksarembededhyarestrednthndextabe thus allowing for fastretrieval,

## Safety filtering

toChapter4.

## Query expansion

chancesofretrievingmorerelevantresults.

To learn more about query expansion and its technical details,refer to[37].

## Retrieval

The retrieval component is responsible for finding the data chunks that are most relevant to the user's query. The user query isfirst converted into an embedding using the CLIp textencoder,and then an ANN algorithm is used to efficientlyretrieve themost similar data chunksin theindextable.

## Generation

Once the relevant data chunks are retrieved, the generation component produces the final output.This involves two main steps:

- ·Prompt Engineering:Theuser query and retrieved context are combined into a prompt and then optimized using techniques such as CoT to structure the model'sreasoning process.
- LLM: The LLM generates the final response using top-p sampling.

## Other Talking Points

- Tabular detection in document parsing[38,39,40].
- Details of approximate nearest neighbor algorithms[20,21,23,24].
- ·Support user-uploaded documents [2].
- ·Dynamicretrieval strategy[41,42].
- ·Query rewriting and expansion [43,37].

- Inference timeCoT and test-time scaling [30,31].

Other Talking Points | 221

## Summary

<!-- image -->

## Realistic Face Generation

## Introduction

This can be useful in entertainment, marketing, and virtual reality. In this chapter, we explore the technologies behind face generation.

Figure 7.1: Realistic faces generated by StyleGAN2 [1]

<!-- image -->

## Clarifying Requirements

Here is a typical interaction between a candidate and an interviewer:

Candidate:What is the primary application of the face generation system? Interviewer:Theinitial focus is on entertainment and content creation,butwe would also consider using it for data collection in the future.

Candidate: Is the focus only on faces? Or does the entire body need to be generated? Interviewer:Let's focus on faces only.

Candidate: Should the generated faces represent a diverse range of ethnicities, ages, and genders?

Interviewer: Yes. This is crucial to ensure inclusivity and avoid biases.

Candidate; Should the system allow control over facial atributes? For example, editing

thefacial expression of a generated imagewhile preserving its identity? optionally discuss attribute control.

dataset.

Candidate:What is the desired image resolution?

Interviewer:Let'saimfor1024x1024.

Candidate:What is the expectedspeed togenerate aface?

Interviewer:The system should generate faces in near real-time -less than a second.

## Frame the Problem as an ML Task

## Specifying the system's input and output

Ina face generation system,users usually don't provide specificinputs;they simply request the generation of a new face. Since machine learning(ML) models need numerical inputs to start,most image generation models begin with a random noise vector.This noise serves as the initial input,which the model then transforms into a realistic image. lf the system supports attribute control,users can also provide desired attributes as input to guide the generation.

The output,which isgenerated inresponse to therandom noise input,is arealistic image of a human face. This output should also reflect the desired attributes (if specified),such as age,gender, and hairstyle.

Figure 7.2:Input and output of a face generation system

<!-- image -->

## Choosing a suitable ML approach

- .Variational autoencoder
- .Generative adversarial network
- .Autoregressive model
- ·Diffusion model

## Variational autoencoder

this learned distribution.

A VAE consists of two main components:

- .Encoder
- ·Decoder

representation of the input image.

Decoder: The decoder is another neural network that maps the encoded representation into an image. The output of the decoder is an image of the same size as the original input image.

During training, the VAE encodes the input into a latent space and then reconstructs the original input from this encoded representation. After training, the VAE can generate new images by sampling points from the learned distribution and using the decoder to map these points into image form.

Through a reparameterization trick(see[2] formore details),VAEsmodel the latent vector as being sampled from a multivariate Gaussian distribution.The modeling of latent space helps theVAE learn meaningful representations thatcan be smoothly interpolated,which is advantageous for tasks such as image morphing and creating variations of input data.

Figure 7.3:VAE training and inference process

<!-- image -->

VAEs have several strengths and weaknesses.

## Pros:

- ·Simple architecture: The encoder and decoder are neural network architectures that are simple to implement.
- Fast generation: Compared to other approaches, VAEs offer fast image generation. into an image using the decoder.
- Stable training:Traininga VAE is typically easy and stable.
- compressing images into lower-dimensional representations.

## Cons:

230|Chapter7.RealisticFace Generation that lack sharp details.

- or artificially generated.

differentiate from real ones.

Figure7.4:GAN trainingand inferenceprocess

<!-- image -->

## Pros:

- High-quality generation: GANs are known for their ability to generate high-quality images.
- ·Fast generation: While GANs are generally slower than VAEs, the generator can stll generate an image in a single forward pass.
- old.

## Cons:

- [5],where the GAN model fails to stabilize during training.
- that, for example,using a text description to generate an image [6].

232 | Chapter 7.Realistic Face Generation

their training data.

editing.

## Autoregressivemodel

capture long-range dependencies.

Figure 7.5:Autoregressive model training and inference process

<!-- image -->

## Pros:

- ·High detail and realism: Autoregressive models generate images with high levels of detail and sharpness,
- ·Stable training: Compared to GANs, training autoregressive models is usually more

stable.

- partoftheinputsequence.
- asa sequenceofnumericalvectors.
- For example, they can generate an image of "an avocado on a chair on Mars"" even if theyhaven't seensuch examples in their training data.

## Cons:

- ·Slow generation:Autoregressivemodels generate the image sequentially,one token at a time.This sequential generation makes them slower compared to VAEs or GANs.
- ·Resource-intensive: These models are usually very large, with billions of parameters. Training such large models requires significant computational resources,which increases the cost.
- Limited image manipulation:Unlike VAEs and GANs,autoregressive models don't have a structured latent space that can be easily explored ormanipulated.This limits certain types of image manipulations such as attribute control in faces.

In summary,while autoregressive models are slow in generation due to their sequential nature, they can generate highly detailed and novel images. Many popular image generation models, such as OpenAl's DALL-E [7] and Google's Muse [8], are based on autoregressive modeling. Chapter 8will examine this approach in more detail.

## Diffusion model

number of steps,with the model adding details to the image at each step.

Figure 7.6: Diffusion model training and inference process

<!-- image -->

## Pros:

- High detail and realism: Diffusion models can generate images of exceptional quality and realism.
- ·Stable training: Compared to GANs, training diffusion models are typically stable.
- ·Control over generation: Similar to autoregressive models, diffusion models can be controlled using various inputs, such as text describing the desired image.
- ·Novelty and creativity: Diffusion models can generate novel and imaginative images.
- ·Robustness to noisy images: Diffusion models are effective at removing noise from images because of their denoising process. This can be useful in certain applications such as image denoising.

## Cons:

- ·Slow generation: Diffusion models generate images in multiple denoising steps. This iterative processmakes them slower compared to other methods.
- ·Resource-intensive: Diffusion models are usually large,with billions of parameters. This makes them computationally intensive and,therefore,expensive to train.
- ·Limited image manipulation: Unlike VAEs and GANs,diffusion models don't have a structured latent space for manipulating images.

examine diffusion models ingreatdetail.

Table 7.1:A comparison of different imagegeneration approaches

| Characteristics        | VAE      | GAN      | Autoregressive   | Diffusion   |
|------------------------|----------|----------|------------------|-------------|
| Quality                | Low      | Moderate | High             | Exceptional |
| Speed                  | Fast     | Fast     | Slow             | Slow        |
| Training stability     | Stable   | Unstable | Stable           | Stable      |
| Control overgeneration | Limited  | Limited  | Flexible         | Moderate    |
| Facial manipulation    | No       | Yes      | No               | No          |
| Novelty                | Limited  | Limited  | High             | High        |
| Resource intensity     | Moderate | Moderate | High             | High        |

Forrealisticfacegeneration,weselect GANsas ourprimary approach.GANs areparticularlyeffectivebecausetheyletusmanipulatefacial attributes throughastructured latent space,which is an optional requirement in thischapter.

Figure 7.7: Facial attribute manipulation (Images are taken from [10])

<!-- image -->

## Data Preparation

applythefollowingsteps:

- from high-quality images.
- afterward.
- .Normalize and resize images:We resize alimages to a standard dimension, for examand1.
- Enhance diversity:WeuseML classifiers to tag imageswith genderage,and other attributes. Then, we adjust the dataset to ensure a balanced representation of different groups. This step is crucial to prevent biases in generated faces.

## Model Development

## Architecture

GANs consist of two components: a generator and a discriminator. Let's briefly examine the architecture ofeach component.

## Generator

The generator component takes random noise as input and converts it into an image. Its architecture consists of a series of upsampling blocks,each of which increases the spatial dimensions (height and width) of its input. These blocks gradually transform the lowdimensional noise vector into a 2D image of the desired size.

Figure 7.8:Series of upsamplingblocks

<!-- image -->

Let'stalkaboutthethreemain componentsof theupsamplingblock:

- Transposed convolution
- ·Normalization layer
- ·Non-linear activation

## Transposed convolution

Transposed convolution,also known as deconvolution or upsampling convolution,is an operation used inneural networks to increase the spatial resolution offeature mapsessentiallyperforming the opposite ofaregular convolution.It'swidely used in applications such as image generation,semantic segmentation,and super-resolution where the goal is toreconstruct higher-resolution outputs from lower-resolutioninputs.

Unlike standard convolution, which slides a filter across the input, transposed convolution starts by inserting zeroes between the pixels of the input feature map, effectively expanding it. The expanded input is then convolved with a filter, where the flter's stride and padding2 are adjusted to achieve the desired output size. For example, starting with Figure 7.8.

pixels

238 | Chapter 7.RealisticFace Generation

Figure 7.9: Transpose conv with 3x3 filter and stride=1 over 4x4 inputs [11]

<!-- image -->

<!-- image -->

<!-- image -->

## Normalization layer

consistent distribution.

criminator competing against each other. This can lead to problems like mode collapse, where the generator produces limited diversity, or oscillations, where the generator and discriminator fail to converge during training. Normalization helps stabilize the training by scaling the activations at each layer, thus reducing the risk of vanishing or exploding gradients. This helps maintain consistent distributions of activations, which is critical for balanced competition between the generator and discriminator.With a more robust optimization process,we can useahigher learning rate, speed up training,and reduce the time needed for convergence. We will discuss the training challenges and mitigations later in this chapter.

There areseveral normalization layers,eachwith itsown way of normalizingdata:

- ·Batchnormalization
- ·Layer normalization
- ·Instancenormalization
- ·Group normalization

## Batch Normalization(BN)

BN [12] normalizes the inputs of a layer across the batch dimension by calculating the mean and variance for each feature.The normalized data is then scaled and shifted using learnableparameters.

- ·Benefts: BN helps stabilize the learning process and allows for higher learning rates, speeding up training. It also acts as a regularizer, reducing chances of overftting.
- ·Usage: Commonly used in deep networks, including Convolutional Neural Networks

<!-- image -->

(CNNs) and GAN generators.

## Layer Normalization (LN)

ture across the entirefeaturevectorofeachsample.

- RecurrentNeural Networks(RNNs) and Transformers.
- acrosssamplesiscrucial.

## Instance Normalization(IN)

IN [14] operates by normalizing across each feature map individually for each sample.

- ·Benefits: IN is beneficial for tasks where the appearance of individual samples varies widely,asitallowsthenetworktofocusoncontentratherthanstyle.
- ·Usage: Commonly used in style transfer and image generation tasks.

## Group Normalization(GN)

GN[15] normalizes inputsby dividing features into groups and normalizingwithin each group. It offers a balance between BN and LN.

- ·Benefits: GN is useful for cases with very small batch sizeswhere BN might not be effective.
- Usage:Often applied in tasks where BN fails due to small batch sizes or when layer behavior consistency isneeded acrossgroupsoffeatures.

<!-- image -->

## Non-linear activation

tions.

Figure 7.11:Generatorarchitecture

<!-- image -->

## Discriminator

The discriminator's job is to differentiate between real and generated images. It functions as a binary classifer, taking an image as input and outputting the probability that theimage is real.

The discriminator comprises a series of downsampling blocks followed by a classification head. The downsampling blocks progressively reduce the spatial dimensions of the input image while extracting its features.The classification head then processes the extracted features to predict theprobability that the input image is real.

Figure 7.12:Seriesof downsamplingblocks

<!-- image -->

A downsampling block consists of several convolution operations to progressively decrease the input''s spatial dimension. In PyTorch,we typically use a "Conv2D" layer with a toenhance trainingstability and performance.

1,which is crucial for interpreting the output as a probability.

Figure 7.13:Discriminator architecture

<!-- image -->

Various versions of GANshavebeen developed over the years toserve different purposes. Forinstance,StyleGAN[18] modifiesthegenerator'sarchitecturetocontrol attributesof generated faces,such as age,hair color,and facial expression. For more details on the StyleGAN architecture and itskey architectural choices,refer to[18].

## Training

To produce realistic images, we train the GAN using a unique process called adversarial training.In adversarial training,the generator and the discriminator are trained simultawhile the discriminator improves its ability to distinguish between real and generated images. During this adversarial process,the generator learns to produce increasingly more convincing images.At the same time, the discriminator gets better at identifying fake images. This competitive process continues until the generator produces images that the discriminatorcannolongerdetectasfake.

prove together, avoiding scenarios where one dominates the other. Such a balance is emto alternatebetween the following two steps:

- 1.Train the discriminator for a few iterations while keeping the generator frozen.
- 2.Train the generator for a few iterations while keeping the discriminator frozen.

242 |Chapter7.RealisticFaceGeneration

Figure 7.14:Generator and discriminator alternative training

<!-- image -->

Next, let's examine the ML objective and loss function for training a GAN model.

## ML objective and loss function

into a unified loss function for the GAN.

## Discriminator

Where:

<!-- formula-not-decoded -->

- D(x) is the discriminator's predicted probabilities fora real image,
- G(z)) is the generator's output (a fake image) given random noise,
- ·m is thenumber ofreal images,
- ·n is the number of fake images.

## Generator

lated as maximizing log (D (G (z) for all fake images or, equivalently, minimizing the followinglossfunction:

<!-- formula-not-decoded -->

## GAN'sminimaxloss

The minimax loss[19],originally used in the GAN paper,unifies the generator's and discriminator'slosses into a single function:

<!-- formula-not-decoded -->

The discriminator aims to maximize the loss, while the generator aims to minimize it. Therefore,the overall ML objective is:

<!-- formula-not-decoded -->

ing stability in GANs. To learn more about these loss functions, refer to [20].

## Common training challenges in GANs

section,we discuss three main challenges of training GANs:

- ·Vanishinggradients
- ·Mode collapse
- ·Failure to converge

244|Chapter7.RealisticFaceGeneration

## Vanishing gradients

tor's learning process.

- ·Modified minimax loss
- Wasserstein loss

thegenerator'sobjectivetomaximize

## Mode collapse

<!-- formula-not-decoded -->

This minor adjustment is inspired by formulating the ML objective from a different perspective.With this change, the generator aims to maximize the probability of fake images being identified as real, rather than minimizing the probability of fake images being identified as fake.

<!-- formula-not-decoded -->

Generator loss:D(G(z))

"mode collapse" Two common techniques to mitigate mode collapse are:

- Wassersteinloss
- Unrolled GAN[22]

[4].

## Failure to converge

happens.

As the generator improves during training, the discriminator's performance declines because it becomes increasingly more difficult to distinguish between real and fake images. If the generator reaches a point where it can mimicreal data perfectly,the discriminator's accuracy drops to 50 percent; it effectively begins to make random guesses, much like flipping a coin. This decline in discriminator performance hinders GAN convergence since its feedback gradually becomes less and less useful to the generator. When training continues past a certain point, the generator starts training on useless feedback, and its qualitymaycollapse asaresult.

There are various approaches to improve the training stability and convergence of GANs:

- ·Normalization: Applying techniques like batch normalization helps stablize training by ensuring consistent distributions across layers.
- inator canhelp balance their progress and avoid instability in training.
- ting and helpsmaintain training stability.
- competition between the generator and the discriminator.

## Sampling

246|Chapter 7.RealisticFace Generation To generate a realistic face image, we sample a point from this latent space, known as a latent vector. The generator then takes this latent vector and transforms it into an image.

Figure7.15:Latent spacepointsmap to face images

<!-- image -->

There are two methods to sample a latent vector from a learned latent space:

- ·Random sampling
- ·Truncated sampling

## Random sampling

of varied images.

## Truncated sampling

Figure 7.16: Random vs. truncated sampling (Gray regions represent sampling areas)

<!-- image -->

<!-- image -->

In summary,random sampling ensures diversity by exploring the entire latent space,while truncated sampling focuses on a high-probabilityregion to enhance realism.Forrealistic facegeneration,weutilizerandomsamplingasit leadstodiversityandusuallyworkswell in practice.

## Evaluation

## Offline evaluation metrics

Evaluating image generation systems involves assessing both the quality and diversity of the generated images. Several metrics have been developed for this purpose, such as Inception score [25], Frechet Inception distance (FID) [26], and Kernel Inception distance (KID) [27]. Among these, Inception score and FID are the most widely used. Human evaluation also continues to be an essential method of assessing generative models.

Let'sbriefly examine the Inception score and FID.

## Inceptionscore

world objects,

Here's a step-by-step explanation of how the metric is calculated:

- want to evaluate.

248 |Chapter 7.RealisticFaceGeneration

- that the model recognizes it as a clear instance of a class.
- thepredictedclass probabilitiesacross allmagesThishelpsuunderstand theverall distribution of classes represented in the generated set.If the images are diverse,the marginal distribution will be flat and spread across many classes.
- 4.Computing KL divergence:TheKL divergence measures how different the predicted class distribution for each image is from the marginal distribution. High-quality images will have a distribution that is very different from the marginal distribution. This is because a high-quality image is expected to have a peak in its distribution,while the marginal distribution is expected to be close to uniform if the images are diverse.
- 5.Calculating the Inception score: The Inception score is the exponentiated average of the KL divergence across all images. A high Inception score indicates that individual images have been confidently classified into a variety of classes,which means the generated images are both diverse and of high quality.

<!-- image -->

How does the Inception score measure both diversity and quality?

- images are spread evenly across differentclasses.

## Frechet inception distance (FiD)

FID is another popular metric for evaluating the quality of images produced by generative models. It assesses how similar the distribution of generated images is to the distribution of real images. Unlike the Inception score, which uses class probabilities, FID considers the statistics of the features extracted by a pretrained model such as Inception v3. The Inception model is chosen because it is trained on a large and diverse dataset (ImageNet) and can extract meaningful features that represent the content and style of the images.

Here's a step-by-step explanation of how FID is calculated:

- 1.Generatingimages:We startby generating a large set of images using themodel we wantto evaluate.These imageswill be compared to a setof real images to evaluate their quality and diversity.
- 2.Extractingfeatures:Wepasseach image (both generated and real) through the Inceptionv3modeland extractfeatures("activations")from a specific layer,usually onenear the end of the network. Features from this deep layer capture high-level informationsuch as shapes, textures, and objects-which is crucial for assessing the realism of the images.
3. Calculating mean and covariance: We calculate the mean and covariance of the extracted features separately for generated and real images. These statistical measures summarize the distributions of features for both sets of images.
- learn more about the Frechet distance and its formula,refer to[29].

Figure 7.18:FID calculation

<!-- image -->

## How does the FID measure both diversity and quality?

- Diversity: The FID considers the covariance of the features, reflecting the spread and variation in the image features. A diverse set of generated images will have a feature distribution similar to that of real images, showing the model's ability to produce a wide range of different images.
- ·Quality: FID ensures the generated images are high-quality by comparing their feature ages' features are similar to those of the real images, this indicates that the generated images are likely to be of high quality.

ment with human evaluation,

## Human evaluation

based on criteria that align more closelywith human judgment.

Figure 7.19:Pairwise comparison during human evaluation

<!-- image -->

## Online evaluation metrics

Various metrics are typically monitored in practice to ensure that the image generation system performs well andmeets user expectations.Two common metrics are:

- User feedback: This metric is vital as it directly reflects users' opinions about the generated images.Userfeedbackcanbegathered throughsurveys,ratings,ordirectcom ments.
- Latency:Latencyrefers to the time it takesfrom when a request ismade until the image is fully generated and delivered to the user.Fast response times are crucial for maintaining a good user experience,especially in interactive applications.Monitoring latency helps identify performance bottlenecks and ensures the system meets user expectations.

## Overall ML System Design

key componentswe'll examine are:

- Facegenerator
- ·Training service
- Evaluation service
- Deploymentservice

Figure 7.20:Realistic face generation overall design

<!-- image -->

## Face generator

latent space based on these attributes.

and latent manipulation, read [32,33],

## Training service

## Evaluation service

model meets quality standards and should replace the existing one.

## Deploymentservice

## Other Talking Points

- offsofeacharchitecture[34,35,18].
- methods[36].
- ·Utilize conditional GANs (cGANs) to generate faces based on specific conditions or inputs[37].
- Evaluation metrics of condition consistency[38,39].
- ·Style-mixing in face generation[18].

## Summary

Clarifyingrequirements

FramingasML

MLapproach

Specifyinginput and output

VAE

GAN

Autoregressive model

Diffusionmodel

Removinglow-quality andlow-resimages

Augmentimages

Normalize andresize images

Enhance diversity

Architecture

Model development

Offline

Online

Overall system components

Other talking points Training Generator Discriminator Adversarialtraining Minimax loss function Vanishing gradients Challenges Mode collapse Failure to converge Random sampling Sampling Truncated sampling Inception score FID

Human evaluation

User feedback

Latency

Face generator

Training service

Evaluation service

Deployment service

Summary Data preparation Evaluation TransposedConv Normalization Non-linear activation BN

LN

IN

GN

Summary|255

## High-Resolution Image Synthesis

## Introduction

this chapter, we explore a technique that enables the generation of detailed and varied images in just seconds.

Figure 8.1:An image generated by VQGAN[1]

<!-- image -->

## Clarifying Requirements

Here is a typical interaction between a candidate and an interviewer:

Candidate: Should the system focus on specific categories of images at the start? Interviewer: For simplicity, let's begin with natural scenes and urban landscapes. We can explore other categories later.

Candidate: Do we have training data consisting of natural scenes? What's the dataset size?

Interviewer: We have a large dataset with about 5 million high-resolution images of natural scenes and landscapes, ingthe desired image?

However,the system should beflexible to support input prompts.

Candidate: What resolution range should we aim for when generating the images? pixels,based on userrequests.

Candidate: Should the images be generated in real time, or is some delay acceptable? is important.Let's aim forfivesecondsperimage.

## Frame the Problem as an ML Task

## Specifying the system's input and output

For high-resolution image synthesis, the user simply requests a new image. The output is a high-resolution image.

Figure 8.2:Input and outputof an image generation system

<!-- image -->

## Choosing a suitable ML approach

onebestsuited forthetask.

tion process,reducing the diversity of the images.

demand.

components.

Figure 8.3:Autoregressive image generation

<!-- image -->

11.

Thisapproach relieson twoprimarycomponents:

- ·lmage tokenizer
- ·Image generator

## Imagetokenizer

This is crucial in autoregressive models, where the image is generated sequentially, chunk by chunk.

The image tokenizer is a separate model, trained independently. Its main functions are to encode an image into a sequence of discrete tokens and decode a sequence of discrete tokensbackintoanimage.

## Image tokenizer(encoding)

## Image tokenizer (decoding)

Figure 8.4:Image tokenizer's encoding and decoding

<!-- image -->

## Image generator

tokens as output,which are then decoded into an image.

262| Chapter 8.High-Resolution Image Synthesis

Figure 8.5: Decoder-only Transformer's flexibility in handling various modalities

<!-- image -->

In summary, we approach image generation with a Transformer-based autoregressive model. First, an image generator (decoder-only Transformer) generates a sequence of discrete tokens. Then, an image tokenizer decodes these tokens into the final image. We will explore the architecture, training, and sampling processes of these components in detail in the model developmentsection.

Frame theProblem as anML Task|263

Figure 8.6:Autoregressive image generation

<!-- image -->

## Data Preparation

The data preparation process involves two crucial steps:

- ·Image cleaning and normalization
- ·Image tokenization

## Image cleaning and normalization

ones are consistent.This is achieved by applying the following operations:

- diverse,high-quality images.
- to 1,to stabilize the training process.

## Imagetokenization

Figure 8.7: Data preparation process

<!-- image -->

sequence of numerical inputs.

## Model Development Architecture

generator.

## Image tokenizer

The image tokenizer model has two functions:

- 1.Encoding an image into a sequence of discrete tokens
- 2.Decoding a sequence of discrete tokens back into an image

A common architecture specifically designed for image tokenization is the Vector-Quantized VAE (VQ-VAE) [2], which is a variant of the standard VAE discussed in Chapter 7.The VQVAEconsistsof three components:

- ·Encoder
- ·Quantizer
- ·Decoder

## Encoder

The encoder maps the input image into a lower-dimensional latent space.This component encodes important features of the image into an encoded representation.

The encoder's architecture is a deep convolutional neural network (CNN) with several convolution layers, each followed by a ReLU [3] activation function. These layers process the inputimage and extractvisual features.

ing9features,eachwithcchannels

<!-- image -->

## Quantizer

266|Chapter 8.High-Resolution Image Synthesis

- .Avoidingposteriorcollapse
- Reducing the learning space

## Avoiding posterior collapse

in shaping the output.

## Reducing thelearning space

Continuous vectors are difficult to predict sequentially because they have endless possibilities and small differences. By turning these vectors into discrete tokens, the quantizer simplifies the process by allowing the Transformer to focus on fewer options.

The quantizer uses an internal codebook to convert continuous latent vectors into discrete tokens.This codebook contains learnable embeddings that represent different patterns in the input images. Each embedding acts as a token,represented by an integer from 1 to k.The quantizer replaces each continuous vector with the closest token in the codebook based on Euclidean distance[4].

Figure 8.9:Quantization process

<!-- image -->

Note that the quantizer is an embedding table. Its sole parameter is the codebook, which vector with the closest token in the codebook; therefore,the output is a collection of token IDs.

## Decoder

The decoder converts discrete tokens back into the original image. It typically uses a deep CNN with transposed convolutions (ConvTranspose2d) to gradually transform the representation to the original image size. To learn more about convolutions and transposed convolutions,referto[5].

Figure 8.10:Decoding process

<!-- image -->

## Imagegenerator

tasks,which includes thefollowing components:

- book.
- Transformer's internal representation.
- information,

268 | Chapter 8.High-Resolution Image Synthesis tors.

Figure 8.11:Decoder-only Transformer components

<!-- image -->

## Training

In autoregressive image generation, we have two training stages:

- ·Stage l: Training the image tokenizer
- Stage ll: Training the image generator

## Stage I: Training the image tokenizer

steps:

- tion.
- internal codebook.

Figure 8.12:Image tokenizer trainingprocess

<!-- image -->

Since thequantizer lookup operation lacksawell-defined gradientforbackpropagation, the VQ-VAE paper proposes approximating the gradient by copying it from the decoder input directly to the encoder output.This approach means that only the selected tokens receive gradients from the decoder, while unselected tokens do not receive any gradients.

## Training data

visual patterns.

## ML objective and lossfunction

ically employed during the training process:

- Reconstructionloss
- ·Quantization loss

Where:

Where:

- E(x) is the continuous latent vector produced by the encoder, E, from the input x,
- 2g is the quantized latent vector selected from the codebook Z,
- sg(.) represents the stop-gradient operation that blocks the gradients from flowing through the term.It is used here to prevent the codebook from being updated when optimizing the encoder.

For more details on the quantization loss formula, refer to the VQGAN paper [1].

In practice,usingbothreconstruction loss and quantization loss during trainingworkswell for reconstructing low-resolution images.However,forhigh-resolution images,themodel may still produce artifacts. To improve reconstruction quality at high resolutions, two additional loss functions are typically employed:

- ·Perceptual loss
- ·Adversarial loss

Perceptual loss:Perceptual loss measures the difference between the features of the origas VGG[6]. The formula is:

<!-- formula-not-decoded -->

Where:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- ·denotesthefeaturemapof thelayer,I,fromapretrainedVGGmodel,
- xis the original image,
- ·xis the reconstructed image.

thesedetailsin thereconstructedimages.

Adversarial loss: Adversarial loss is derived from GANs [7], where a discriminator tries to distinguish between real and reconstructed images. This loss is used to measure how well the image reconstructed by the image tokenizer can fool the discriminator. The formula, aswe saw in Chapter7,is:

Adversarial loss =-log（D（x))

Where:

- ·D is thediscriminatornetwork,
- ·xis the reconstructed image.

Thisloss function encourages the model toproducereconstructed images thata trained discriminator cannot distinguish from real images. The VQGAN paper introduced a patchbased version of this loss to reduce unnatural artifacts and improve the realism of the reconstructions.

Overall loss: The overall loss function is often a weighted sum of the individual losses described above. The weights (i) are hyperparameters that need tuning based on specific performancegoals and experiments.

```
Overall loss =Arec x reconstruction loss + Aquant x quantization loss + percxperceptual loss+ adv x adversarial loss
```

the image generator,

## Stage ll:Training the image generator

272|Chapter 8,High-Resolution Image Synthesis visual tokens.

## Sampling

In autoregressive models,generating a new image involves two steps:

- 1.Generating a sequence of discrete tokens
- 2.Decoding discrete tokens into an image

## 1. Generating a sequence of discrete tokens

sive nature of the generation ensures that each token is conditioned on preceding tokens, leading to coherent images. minconi2naat Here is a step-by-step process to autoregressively generate a sequence of tokens:

Figure 8.13:Image generator loss calculation

<!-- image -->

Figure8.14:Generatinga sequenceofdiscrete tokensusingtheimagegenerator

<!-- image -->

- 1.Randomly select a token from the codebook as thefirst token.This initial token acts as a seed for therestof thegenerationprocess.
- 2.Autoregressively generate tokens one by one.This involves:
- a.Passing the current sequence of tokens to the image generator to predict the probabilitydistribution over thecodebook
- b.Selecting the next token using a sampling method such as top-p sampling
- c.Appending the chosen token to the current sequence

This process continues until the entire image is generated. The number of iterations depends on the resolution and size of the desired output image. For example, generating an image of 1024x1024 pixels, with each visual token representing a 64x64 pixel block, requires 256 tokens. The process continues until all 256 tokens are generated. Once the sequence of tokens is complete,it is transformed into an actual image, which is the focus of thenextstep.

## 2. Decoding discrete tokens into an image

decoding functionality of theimage tokenizer.

Figure8.15:Decoding tokens into an image

<!-- image -->

## Evaluation

The evaluation metrics for high-resolution image synthesis are similar to those in Chapter 7. We'll briefly review them in this section without going into detail.

## Offline evaluation metrics

generated images:

- ·Inception score: Measures how similar the generated images are to images of realInception score, refer to [8].
- learn more about FID, refer to [9],

- measure ofwhich modelsproduce morerealisticimagesover time.

latencyandcost.

- image. This metric is important to monitor since users generally expect quick results.
- ·Cost per generation: Calculates the cost to generate an image. This metric depends on factors such as model complexity, resolution, and infrastructure expenses. Monitoring the cost ofgeneration is crucial as it impactsbusinessrevenue.

## Online evaluation metrics

In practice, companies monitor various metrics to assess the system's real-time quality. Commonmetricsinclude:

- Userfeedback:Collects directfeedbackfromusersregardinggeneratedimages.
- ·Periodic surveys: Gathers user opinions on the quality and relevance of generated images.
- ·Subscription rate: Measures how often users subscribe to services or features related to image generation.
- Churn rate:Measures the rate at which users stop using the service.

## Overall ML System Design

Once we are satisfied with the performance of the image generator and image tokenizer models, we can integrate them to construct the image synthesis system. The primary components in a high-resolution image synthesis system are:

- Generation service
- Decoding service
- ·Super-resolution service

Figure 8.16: High-resolution image synthesis overall ML design

<!-- image -->

## Generation service

ator model to produce a sequence of visual tokens.

## Decoding service

The decoding service interacts with the image tokenizer to convert the generated sequence of visual tokens into an image. Note that when we deploy the model, we don't need the encoder in the image tokenizer - it is only used during training.

Separating generation and decoding services is crucial because the image generator and tokenizer are different models with distinct computational needs and latencies. This approach allows each service to scale independently and manage resources effciently.

## Super-resolution service

Super-resolution service uses a pretrained model to increase the resolution of generated images.For example, if the desired resolution is 2048x2048 but the generator produces only 1024x1024, we use a super-resolution model with a 2x upscale factor.

## Other Talking Points

If there's time remaining at the end of the interview, you could explore these additional points:

- Extending autoregressivemodels to support text-based generation[13,14].
- ·Support applications such as image completion and image super-resolution [15].
- Balancing diversity vs.fdelity in sampling,using techniques such as temperature scaling [16].
- Enhancing the stabilitywith adversarial training,gradientclipping,and learningrate scheduling[17,18].
- Using progressive growing and multi-scale architectures to improve image quality and detail [19].
- Creating interactive systems for users to refine and customize generated images [20].

## Summary

Clarifyingrequirements

FramingasML

Data preparation Specifying input and output MLapproach Autoregressive modeling Image cleaningand normalization Image tokenization Architecture Imagegenerator Image tokenizer Image tokenizer

Summary Model development Offline Evaluation Online Overall system components Other talking points Training Image tokenizer Imagegenerator Encoder Quantizer Decoder Decoder-only Transformer Reconstruction loss Quantization loss Perceptual loss Adversarial loss Image generator Sampling Next-token prediction Cross-entropy loss Generating discrete tokens Decoding tokens into an image Inception score FID

Human evaluation

Time to generate an image

Cost per generation

User feedback

Periodic surveys

Subscription rate

Churn rate

Generation service

Decoding service

Super-resolution service Summary|279

## Text-to-Image Generation

## Introduction

[2],andAdobe'sFirefly [3].

Prompt:An illustration of an avocado sitting in a therapist's chair,saying  just feel so empty inside'with a pit-sized hole in its center.The therapist,a spoon,scribbles notes.

<!-- image -->

## Clarifying Requirements

Here is a typical interaction between a candidate and an interviewer:

Candidate:What resolution do we target for generated images? Interviewer:We aim for high-resolution images, specifically 1024x1024 pixels.

Candidate: Should the system support multiple languages for text input or just English? Interviewer:We'llfocusonEnglish initially,but the system's architecture should be adaptableforotherlanguageslater.

Candidate:How large is the dataset for training a text-to-image model? Interviewer:Wehave about5oomillionimagesfromuserassets,mostwithcaptions.

Candidate: How detailed and complex can the text prompts be? Is there a limit to their complexityorlength?

Interviewer:The system should handle detailed textprompts,with amaximum length of 128words.

Candidate: What speed should the system achieve for image generation?

Interviewer: The goal is near-real-time generation.Let's aim for 10 seconds per image.

Candidate:What types of images should the system generate? Are we focusing on a specific domain,like landscapes?

Interviewer:The system should be capable of generating,based on textprompts,awide range ofimages,includingrealisticlandscapes,portraits,and abstractor conceptual art.

Candidate:It's important to ensure the images aren't biased by age,race,or gender.Can I startbyfocusing on those three attributes?

Interviewer:Great point. It's crucial to have a fairsystem.Let's begin by addressing those three attributes.

Candidate:Ethical considerations are crucial. We need flters and checks to avoid generating offensive,inappropriate,or harmful images. Does that sound correct? Interviewer:Yes,that's correct.

## Frame the Problem as an ML Task

## Specifying the system's input and output

The input to the system is a text prompt provided by the user that describes the desired imagehisrompt suallyncudesdetailslikescenebjects,colorsstylesandm tions.

284|Chapter9.Text-to-lmageGeneration

Figure 9.2: Input and output of a text-to-image system (lmage credit: [4])

<!-- image -->

## Choosing a suitable ML approach

Text-to-imagegeneration is a multimodal task thatinvolvesunderstanding text andgeneratinga corresponding image.There are two primary approachesforbuilding text-to-image systems:

- ·Autoregressive models
- ·Diffusion models

Let's briefly review each and choose the one that best suits our needs.

## Autoregressive models

the actual image.

Figure9.3:Autoregressive text-to-imagegeneration

<!-- image -->

Several text-to-image modelshavebeen developed using this approach,such asOpenAl's DALL-E[5] and Google's Muse [6].

## Diffusion models

First introduced in 2019 [7],diffusion models gained mainstream attention about three years later. They use a different approach for text-to-image generation by starting with random noise and gradually transforming it into a clear image based on the text prompt. This process typically involves a text encoder, such as OpenAl's CLIP [8] or Google's T5 [9], which converts the text prompt into an embedding. This embedding captures the meaning of the prompt and guides the diffusion model to generate images that match it.

Figure 9.4: Diffusion-based text-to-image generation1

<!-- image -->

Examples of diffusion-based text-to-image models include Google's Imagen 3 [2], OpenAl's DALL-E 2 [10], and Stability Al's Stable Diffusion [11].

## Diffusion versus autoregressive models

Autoregressive models frame text-to-image generation as a sequence generation task, while diffusionmodels approach it as an iterative refinement process. This key difference in modeling impacts their capabilities.

<!-- image -->

- the Transformer in an autoregressive model generates the sequence of visual tokens, these tokens form the nal image.Diffusionmodelshowever,refine theimage over many steps,addingcomplexity to the implementation.
2. Image quality: Diffusion models have shown better performance in generating highly detailed and realistic images.Their iterative process allows the model to continuously refine and enhance fine details,leading to superior overall realism in the generated images.
3. Flexibility in sampling: Diffusion models are more flexible in trading off sampling speed and image quality. They can easily adjust the number of sampling steps-more steps usually lead to higher-quality images but take more time. Once trained, an autoregressive model cannoteasily make such adjustments.

In this chapter,we choose diffusionmodels to prioritize exceptional image quality.In the model development section,we explore the architecture,training methods,and sampling techniquesofdiffusionmodels.

## Data Preparation

Our dataset consists of approximately 5o0 million image-caption pairs.However,largescale datasets often require substantial preprocessing before they can be used for model training. In this section, we explore common techniques to prepare images and captions for diffusion training.

## Image preparation

We focus on two main steps to prepare the images: filtering inappropriate images and standardizing the remaining ones. Let's delve into each step in more detail.

## Filtering inappropriate images

In large-scale datasetsmany imagesmaynot beuseful for training.It's crucial toremove these to ensure themodel learns only from high-quality and safe dataTo achieve thiswe perform thefollowing steps:

288 |Chapter9.Text-to-lmage Generation

- during training.

## Standardizing images

- . Adjust image dimensions: The diffusion model requires inputs of specific dimensions. Therefore,it's crucial to have training data of similar sizes.For example, if the expected model input is 128x128, we first resize the images so that the smaller dimension is 128,preserving the aspect ratio.Then,we center-crop them to achieve the final size of128x128.
- Normalize images:We normalize pixel values to a standard range such as[0,1] or[-1, 1] formore stable training.

Figure9.6:Imagepreparation steps

<!-- image -->

## Caption preparation

ensure captions are consistent and of high quality:

- system from scratch, refer to Chapter 5.

## Model Development

## Architecture

tecturesare typicallyusedforthispurpose:

- U-Net
- Diffusion Transformer(DiT)

Figure 9.7:Input and output of the difusionmodel ina single step

<!-- image -->

## U-Net

by upsampling blocks,as shown in Figure 9.8.

in the training section.

Figure 9.8: U-Net downsampling and upsampling blocks

<!-- image -->

## Downsamplingblocks

The downsampling blocks progressively reduce spatial dimensions (heightand width) while increasing depth (number of channels), leading to a compressed representation of the input.Each downsamplingblock typically consists of the following:

- Convolution operation: Extracts visual features from the input.
- Non-linear activation: Introduces non-linearity to learn complex patterns.
- Batch normalization: Normalizes feature maps to stabilize training.
- Max-pooling: Reduces the feature map dimensions.

Figure 9.9:Typical layers in a downsampling block

<!-- image -->

derived from the image features,while thekeys andvalues come from the text embed into the image features.

## Upsamplingblocks

The upsampling blocks symmetrically increase spatial dimensions and decrease feature map depth.The fnal output matches the original input size,which,in thiscase,is the predictednoise.Each upsamplingblockconsistsof thefollowing:

- Transposed convolution:Uses operations like PyTorch's ConvTranspose2D to process and increase the feature map's dimensions.
- Batch normalization:Normalizes feature maps to stabilize training.
- ·Non-linear activation:Introducesnon-linearity to learn complexpatterns.
- Cross-attention: Maintains the influence of additional conditions during upsampling.

Figure9.10:Typical layers in an upsamplingblock

<!-- image -->

The U-Net architecturehas various details and variations.Different implementationsmay structureisgenerallysuffcientformostMsystemdesigninterviews.Formoreindepth information,referto[14].

## DiT

DiT[15]isanother popular architecture in diffusion models.UnlikeUNet,whichuses

292|Chapter9.Text-to-ImageGeneration Chapter 5. DiT components are:

Figure 9.11: DiT components

<!-- image -->

- cate its location in the original image.
- same dimension as the original input image.

chapter.Chapter 11 explores the DiT architecture in more detail.

## Training

twophases:

- Forwardprocess
- Backwardprocess

## Forwardprocess

orparameterupdates.

Figure9.12:Forward diffusionprocess

<!-- image -->

## Backwardprocess

In the backward process, also known as the denoising process, an ML model learns to reverse the forward process. At each step, the model predicts the noise in the noisy image. 9.13,this process is repeated until the image becomes clear.

294 |Chapter9.Text-to-Image Generation

Figure9.13:Backward diffusionprocess

<!-- image -->

With an understandingof both theforward andbackward processes,wecannow explore how theyare applied duringthe diffusion trainingprocess.

## Diffusion trainingprocess

Duringtraining,we introducenoiseto the originalimagebysimulatingtheforwardprocess, then ask the model to predict this noise.This process involves four key steps:

- 1.Noise addition
- 2.Preparation of conditioning signals
- 3.Noise prediction
4. ML objective and loss calculation

the design of the ML system.

## 1.Noise addition

time.

<!-- formula-not-decoded -->

process.

noiseaddition formula:

where:

- xisthenoisyimageattimestept,
- x-1is thenoisyimageattimestept-1,
- ·e is the Gaussian noise sampled from the standard normal distribution N(0, I),
- added.

Iteratively adding noise over many steps can be time-consuming. Instead, it can be shown that the noisy data at timestep t can be directly derived from the original data, xo:

<!-- formula-not-decoded -->

where:

- ·x is the noisy image at timestep t,
- α = 1 - βt and α = IIi=1 α are reparameterizations of βt,
- ·e is the Gaussian noise sampled from the standard normal distribution N(0, I).

In summary, during noise addition, we randomly sample t and compute x, directly from x,

<!-- formula-not-decoded -->

## 2.Preparation of conditioning signals

be processed by the model.

## 3.Noise prediction

296|Chapter 9,Text-to-Image Generation

<!-- formula-not-decoded -->

x,and the timestep,t.3

## 4.ML objective and losscalculation

and the predicted noise:

<!-- formula-not-decoded -->

## where:

- t is a timestep sampled uniformly from {1, 2,.., T},
- ·~N(o,I) is the Gaussian noise used in theforward process,
- ·x is the noisy data at timestep t, computed using the noise addition formula,
- ·E(x,t) is the prediction of the neural networkmodel (U-Net or DiT).

Figure 9.14: A single iteration in diffusion training

<!-- image -->

diffusion training,refer to[18].

## Sampling

coherentimagesguidedbythetextprompt.

each step producing a clearer image until a clear and detailed image is achieved.

Figure9.15:Samplingprocess

<!-- image -->

Thebasicsamplingprocess described above hastwo drawbacks.First,it oftenfails to generateimagesthataccuratelymatchthetextprompt.Second,itisslow,becausegenerating each sample requires many iterative steps. The following two techniques are commonly employed inpractice tomitigate the issues described above:

- aboutDDIM,refer to[20].

models inmore depth,refer to[21,19,20].

298 |Chapter9.Text-to-lmage Generation

and styles.

The most common ones include:

- .Resource-intensive model training
- ·Slowimagegeneration

## Resource-intensive model training

Training diffusion models is computationally intensive, requiring significant processing power. It also requires substantial GPU memory due to the size of themodel and the high-dimensional nature of the generated images. Most modern GPUs may not have sufficient memory to fit model parameters,activations,and gradients during training.Thefollowing strategies are commonly employed to overcome these challenges:

- Mixed precision training: This technique uses both 16-bit and 32-bit floating-point types to reduce memory usage and increase computational efficiency. To learn more, refer to [22].
- Model and data parallelism: These methods distribute training across multiple devices. Most distributed training frameworks, such as FSDP [23] and DeepSpeed [24], support various parallelism techniques.
- this approach in more detail.

## Slow image generation

putations occur at each step.

- needed to generate images [25].

## Evaluation

## Offline evaluationmetrics

evaluation to assess threekeyareasof themodel's ability togenerate images:

- ·Imagequality
- Imagediversity
- Image-textalignment

As discussed in previous chapters, Inception score (Is) [27] and Frechet Inception distance (FID) [28] are two common metricsforevaluatingquality and diversity in image generation systems.In this section,we focus primarily on image-text alignment.

## Image-text alignment

Image-text alignment refers to how accurately the generated images match the text prompts. Measuring this alignment isimportantbecause it ensures the generated images arefaithful to the user'sinput.A commonmetricfor assessing this isCLiPScore[29],which evaluates the degreeof alignment.Before diving into CLiPScore,let'sbrieflyreviewCLiP.

## CLIP

CLIP [8] is a model developed by OpenAl that has been trained to match images with their corresponding descriptions.It consists of two encoders:one for text and one for images. The text encoder converts input text into a text embedding; the image encoder converts the image into an image embedding.

Figure9.16:CLIPencoders

<!-- image -->

During training,CLiP learns to align embeddings by bringingrelated text and image embeddings closer together and pushing unrelated ones apart. This helps CLiP develop a shared embedding space where both an image and its associated text will map into the same space.

After training, similar text descriptions map close to each other in the embedding space, and images map near their relevant descriptions.

<!-- image -->

## CLIPScore

represents the embedding size, Evaluation|301

age and a description.

## Human evaluation

and text alignment using thefollowingapproach:

- what," and "no" are scored as 100, 50, and 0, respectively. These scores are averaged separately for generated images and reference images to measure text alignment.

## Online evaluation metrics

Online metricsmeasure how the model works in production. Common metrics to evaluate ourtext-to-imagemodel include:

- Click-through rate (CTR):The percentage of users who click on the generated images. A high CTR indicates that usersfind the generated images useful.
- times indicate higher user engagement.
- Userfeedback:Direct feedback from users is collected through feedback. Positive feedbackindicatessatisfactionwith the image quality and text alignment.
- Conversion rate: The percentage of users who take a desired action (e.g., purchase, sign up) after interacting with generated images. A high conversion rate indicates satisfactionwith themodel'sperformance.
- ·Latency: The time it takes to generate an image from a text prompt. Lower latency indicatesfaster performance,which isimportantfor usersatisfaction.
- High throughput ensures the service can serve more users.
- that costs remain justifiable.

302 |Chapter9.Text-to-lmage Generation

## Overall ML System Design

pipelines:

- .Datapipeline
- .Training pipeline
- ·Modeloptimizationpipeline
- ·Inference pipeline

## Data pipeline

The data pipeline prepares data for training byremoving inappropriate images,standardizing the rest, and storing them. It ensures captions are present and relevant and uses a pretrained model such as Google's T5 [9] to pre-compute and cache caption embeddings. This caching reduces computation during training.

In adition to preparing text-image pairs from the training data, the pipeline also collects feedback. This new data is added to the training set for future use.

Figure 9.18:Data pipeline

<!-- image -->

## Training pipeline

pipeline.

Figure9.19:Trainingpipeline

<!-- image -->

The trainingpipeline ensures that the model adapts to recent user prompts and is trained onhigher-quality generated images.

## Evaluation pipeline

The evaluation pipeline assesses newly trained models using predefined automated metrics to determine whether or not they meet performance and quality standards for deployment.

## Model optimization pipeline

There are several methods to optimizemodels:

- size and generation time.
- generation time.
- eration,

model in production,

## Inference pipeline

- .Promptauto-complete
- .Prompt safety service
- .Prompt enhancement
- ·Image generation
- ·Harm detection
- ·Super-resolution service

Figure 9.20:Inference pipeline

<!-- image -->

## Prompt auto-complete

Figure 9.21: Suggested phrases by the auto-complete

<!-- image -->

## Prompt safety service

ation ofinappropriateimages.

## Promptenhancement

herence,and details.

Figure9.22:An example of prompt enhancement

<!-- image -->

Thiscomponent iswidely used in advanced image andvideogenerationsystems[30],as it effectively helps the model producebetter outputs.It improvesthe quality of generated images by offering the model a more coherent and detailed prompt.

## Image generation

The image generation component is the core of the inference pipeline. It interacts with the T5 text encoder to encode the enhanced text prompt into a sequence of tokens. These tokens are passed to the diffusion model to generate one or multiple images for each prompt.

## Harm detection

This component ensures that generated images are safe for users. If an image stil contains being shown.

## Super-resolution service

306|Chapter9.Text-to-lmage Generation the desired resolution.

Figure 9.23: A cascade of super-resolution models

<!-- image -->

In summary, various pipelines work together to ensure that a text-to-image system is reliable, high-quality, and safe. The data pipeline provides the foundation for continuous improvement, while the training and model optimization pipelines enhance the model's performance. The inference pipeline ensures safe, efficient, and high-quality image generation.These pipelines create a systemready for real-world challenges.

## Other Talking Points

If time permits at the end of the interview, consider discussing these additional topics:

- Using consistency models for faster image generation [26].
- Employing RLHF for quality improvement [32].
- [33].
- Pack[35].
- [38].,

<!-- image -->

## Personalized Headshot Generation

## Introduction

Personalized text-to-image(T2l) models are one of theemerging applicationsof generative Al. Imagine asking a T2l model to generate an image of your friend John with the prompt "John sitting on a chair while reading a book." While the model willikely produce an image of a person sitting and reading, it probably won't depict "John" To do that, we need to personalize a T2l model by having it learn the subject of interest (i.e, John).

Generated images in novel poses and styles

<!-- image -->

Candidate: How many headshot images should the system generate?

Interviewer:50images,

<!-- image -->

<!-- image -->

Al-generated headshots

Figure 10.2: Input and output of a headshot generation system

## Choosing a suitable ML approach

range of images as a base model.

and tuning-free.

Frame the Problem as an ML Task| 315

<!-- image -->

- ·Textual inversion
- ·DreamBooth
- ·Low-rankadaptation(LoRA)

## Textual inversion

Textual inversion [4] personalizes a T2l model by introducing a new special token that the special token's embedding, while the diffusion model, text encoder, and other token embeddingsremainunchanged.

<!-- image -->

special token.

Frame the Problem as an ML Task| 317

Figure10.6:Generating animageof the subjectof interest

<!-- image -->

Let'sreview thepros and cons of textual inversion.

## Pros:

- Efficiency: Textual inversion training involves learning only a new token embedding, makingit a lightweight and efficientprocess.
- Preservation of original model capabilities:The diffusion model's capabilities are maintainedsince itsparametersremain unchanged.
- Minimal storage requirements:Minimal storage is required because only the special tokenembeddingneedstobestoredforeachpersonalizedmodel.

## Cons:

- ·Difficulty in learning subject details: Textual inversion often struggles to learn new subject details accurately due to its limited capacity to encode these details. This limitation arises because the new subject is represented by a single token embedding.

it often struggles to capture and preserve all the details of the new subject.

## DreamBooth

318| Chapter 10.Personalized Headshot Generation DreamBooth relies primarily on two techniques for successful finetuning:

Figure 10.7:DreamBooth method updating the entire diffusion model

<!-- image -->

- ·Rare-token identifier
- Class-specific prior preservation loss

## Rare-token identifier

ispreferred, Frame theProblem as an ML Task|319

asa uniqueand cohesiveentity.

## How therare-tokenidentifierworks

thattokenizerstreatthemasasingleunit.

Here is the step-by-step process to form anidentifier:

- large set of tokens, each with a unique ID. A "rare token" is one that appears infrequently in the training data. Such rare tokens are identified by assessing token frequencydistribution.
2. Generating a sequence of rare tokens: Once we have identified the rare tokens, we generate a sequence using some of these tokens.
- 3.Forming the identifier:The tokenizer converts the sequence of token IDs back into their corresponding text form.This forms an identifier to represent the subject."XyZ" "SKS",and"[V]"are possible examples of an identifier.

## Class-specificprior preservation loss

DreamBooth finetunes all layers of the diffusion model. While this enhances the quality of generated images, it can lead to overftting, where the model loses diversity in its outputs. generate images ofother dogs.

We will examine DreamBooth's loss function in the training section.

DreamBooth hasseveral pros and cons.

## Pros:

- learn the details of the subject more accurately.

set is needed to learn the subject.

## Cons:

- is costly and not scalable.
- during training.

LoRA, which offers a balanced approach.

## LoRA

LoRA, introduced by Microsoft [6], is a powerful method for efficiently finetuning very large models. This method was originally developed for adapting large language models (LLMs) to specific tasksbut was later adopted for other tasks including T2l personalization.

The key motivation behind LoRA is that finetuning all parameters of large pretrained models, such as GPT-3 [7], is time-consuming and costly. Instead, LoRA adapts a large model to a new task by introducing a small set of parameters and updating only those, thus significantlyreducing computational costs.

Due to its importance, let's examine in detail the mathematical foundations of LoRA.

## The mathematics of LoRA

<!-- image -->

Figure 10.9: Injection of low-rank matrices

<!-- image -->

## LoRA for T2l personalization

eficient.

Frame the Problem as an ML Task | 323

<!-- image -->

Table 10.1:Comparison ofpopular tuning-based personalization methods

|                                               | Textual Inversion   | LORA     | DreamBooth   |
|-----------------------------------------------|---------------------|----------|--------------|
| Learning effectiveness                        | Low                 | Moderate | High         |
| Required storage                              | Low                 | Moderate | High         |
| Required training resources                   | Low                 | Moderate | High         |
| Maintaining the original model's capabilities | Yes                 | Yes      | No           |

## Which method is more suitable for headshot generation?

The suitability of these methods depends on the use case and the system requirements. For headshot generation, we choose DreamBooth for three main reasons:

1. Better identity preservation: DreamBooth is most effective at preserving details of the subject, leading to better identity preservation.

## Data Preparation

need around 10-20 images.

follow these steps:

- ·Image resizing
- ·Generic face data addition
- ·Image augmentation

Figure10.11:Preparingdatafortraining

<!-- image -->

## Image resizing

Diffusionmodelstypicallyrequire fixed inputdimensions,butuser-uploaded imagesoften vary in dimensions.We resize the images to uniform dimensions suitable for the diffusion model.

## Image augmentation

T2l models require lots of images to learn concepts such as objects, identities, and scenes. However,forpersonalization,we often lack a large dataset.We use image augmentation techniquessuch asmirroring,slightrotations,andscalingto artificiallyexpand the dataset. This step is essential when only a small number of images is available for training.

## Genericface data addition

Training only on the provided images may cause the model to overfit to a specific identity and forget previously learned knowledge. To prevent this,we combine user-uploaded images with a larger,generic dataset of faces.We generate these images using a pretrained diffusion model with prompts such as"an image of a person."

## Model Development

## Architecture

The DreamBooth method finetunes a pretrained diffusion model. We use a model with a U-Net architecture,pretrained to output 1024x1024 images, imilar towhat we covered in by a series of upsamplingblocks.

## Training

model fromscratch:

- using the conditioning signals.

## Training data

data preparation. User-uploaded images are labeled as "An image of a [V] person, while generic face images are labeled as "An image of a person"

## ML objective and lossfunction

The key challenge in personalization is to ensure the model can generate both general categories (e.g.,human faces) and specific instances within those categories (e.g.,a unique individual). To address this,we use two loss functions:

- ·Reconstruction loss
- ·Class-specific prior preservation loss

## Reconstruction loss

This loss function measures the differences between the reconstructed image and the actual images of the specific subject. It helps the model preserve the subject's identity.

## Class-specific prior preservation loss

## Overall loss

Figure 10.12:Overall loss calculation

<!-- image -->

## Sampling

The sampling process for headshot generation is similar to the T2l generation discussed in Chapter9,asboth usediffusion models.However,akey difference lies in howwe provide text prompts. In Chapter 9, the user provided the text prompt. For example, a user might input"a cat sitting on a chair, and the diffusion model would generate an image reflecting thattext.

In headshotgeneration,the users don't provideprompts.nstead,we createa set of hand engineered prompts. These prompts represent various professional settings and include the identifier used during training to ensure the generated images reflect the user's identity. Some examples include:

- ·"A professional headshot of [V] smiling in front of a plain white background."
- ·"A close-up of[V] with a neutral expression,wearing formal attire."
- ·"A headshot of [V] with soft lighting,looking slightly to the left."
- ·"A profile shot of [V] against a blurred outdoor background."

We follow the standard diffusion process sampling steps:

- 1.Generate initial random noise.

328| Chapter 10.Personalized Headshot Generation

- refer to [8] or Chapter 9.

Figure10.13:Samplingaheadshotimage

<!-- image -->

## Evaluation

## Offline evaluation metrics

It is important to evaluate personalized diffusion models to ensure our model is capable of preserving the user's identity in the generated images. We assess the performance of a personalized diffusion model by focusingon three main aspects:

- ·Text alignment
- ·Image quality
- Image alignment

## Text alignment

quality but also relevant to the input text.

## Image quality

## Image alignment

original subject are:

- ·CLIP score
- DINOscore
- Facial similarityscore

## CLIP score

CLIP model [12] uses two encoders-one for images and one for text.These encoders are trained to ensure that the image and text embeddings of arelevant image-text pair are close in theembeddingspace.

Figure 10.14:Image alignmentcalculationwith CLIP

<!-- image -->

To measure image alignment using CLip,we discard the text encoder and use the image encoder to generate embeddings for both the generated and real images. We then calculate the cosine similarity between these embeddings to assess their similarity. Higher scores indicate that the personalized diffusion model creates images that are morevisually similar toreal images.

## DINO score

DINO [13] is a self-supervised learning method developed by Meta. DINO learns visual representationsof images without theneed forlabeled dataIn particular,it uses amethod called contrastive learning [14],whereby the model learns to distinguish between similar and dissimilar images by organizing them in an embedding space-similar images are placed closer together, and dissimilar ones are positioned farther apart.

DINO,longwithitorerecentariations lik215]isparticulalyoodata turing similarities between images because it is trained to recognize subtle differences.

330 |Chapter 10.Personalized Headshot Generation

Figure 10.15:Image alignment calculation with DINOv2[15]

<!-- image -->

## DINOversusCLIP

DINO is preferred for comparing images because it is trained to capture detailed visual might have a low DINO score due to the color difference. On the other hand, CLIP is a person wearing a jacket.

## Facial similarity score

revenue.Wefocus ontwoprimarymetrics:

- indicatethattheheadshotsmeetorexceedexpectationswhilelowerscoreshighliht improvements areneeded.
- Conversionrate to paid service:Thismetricmeasures the percentage ofusers who move from showing interest to becoming new paying customers.It is calculated by dividing the number of new paying customers by the total number of users who engaged with the servicesuch as visiting thewebsite, signing upfora trial, ormaking inquiries-within a specifictimeperiod.

## Overall ML System Design

Generating professional headshots requires more than just a diffusion model. In this section,weexaminethreekeypipelines:

- ·Data pipeline
- ·Training pipeline
- Inference pipeline

## Data pipeline

This pipeline has two responsibilities:

- ·Preparing images with the subject of interest
- ·Preparing images of generic human faces

Figure 10.16:Data pipeline components

<!-- image -->

## Preparing imageswith the subject of interest

This process evaluates user-uploaded images to ensure they meet predefined standards andpreparesthemfortraining.

Specifically, it verifies that the images are diverse and contain only a single object of inter est: the user's face. To achieve this, we use various heuristics and ML models to analyze the diffusion model.

## Training pipeline

Figure 10.17:Finetuning a pretrained diffusion model

<!-- image -->

## Inferencepipeline

The inference pipeline is responsible for generating the headshots of the user using a personalized T2l model. Three main components in the inferencepipeline are:

- Imagegenerator
- Qualityassessmentservice
- ·Uploaderservice

Figure 10.18: Inference pipeline components

<!-- image -->

## Image generator

to generate one image per prompt.

## Quality assessment service

prompt but with a different initial noise.

## Uploader service

## Other Talking Points

- Details of class-specific prior preservation loss[5].
- Addressing the issue ofreduced output diversity afterfinetuning[5].
- Supportingmultiple sizes and aspect ratios in generated images[17].
- Details of tuning-free methods such asMeta's ImagineYourself [1].
- [18].
- suringdataprivacy[19,20].

<!-- image -->

## Text-to-Video Generation

## Introduction

Text-to-video generation is a key application of generative Al,enabling the generation of videos from textual descriptions.This chapter delves into the crucial components required tobuild a text-to-videomodel.

PromptA stylishwomanwalksdown a Tokyo streetflled withwarm glowing neon and animated city signagehewearsablackleatherjacketalongred dressand blackbootsand cariesablack pursehwasunglassesandedlitickhewalksonfidntlyandcasuallheste

<!-- image -->

Candidate:What is the expected length of the generatedvideos?

Interviewer:Let'saim forfive-second-longvideos.

<!-- image -->

## Specifying the system's input and output

prompt.

For example, given a text input like "A dog playing fetch in a park on a sunny day," the system should generate a video depicting this scene, capturing the dogs movement, the park's environment, and the ambiance of a sunny day.

Figure 11.2:Input and output of a text-to-video system

<!-- image -->

## Choosing a suitable ML approach

<!-- image -->

Figure 11.5: Compression network consisting of visual encoder and decoder

<!-- image -->

## How does LDM address computational complexity?

LDMs require less computing power than standard diffusions because processing compressed representations is cheaper than handling high-dimensional pixels. To understand the impact of this compression,let'swalk through an example.

Imagine we need a video with 24 FPs,a duration of five seconds,and a resolution of 720p. This means 120 frames, each with 1280x720 pixels-a substantial amount of data to process. If we use a compression network similar to [4] that reduces both the temporal and spatial resolution by a factor of 8, the video's spatial dimension becomes 160x90 pixels, and its temporal dimension shrinks to 15 frames.

Figure 11.6: Impact of compression on data volume

<!-- image -->

resolution video data, Togenerate avideousinga trained DM,westartwithpurenoise in thelatentspace.The LDM gradually refines it into a denoised latent representation.Thevisual decoder then converts this latent representationback intopixel space to produce the final video.

Figure 11.7: Video generation using a trained LDM

<!-- image -->

For this chapter, we choose an LDM approach to develop our text-to-video generation system because it's efficient and it reduces computational load. To learn more about LDM, referto[6].

## Data Preparation

The dataset for text-to-video generation includes 100 million pairs of textual descriptions and their corresponding videos. These pairs cover various subjects and actions, allowing the model to learn from diverse videos.In this section,we prepare thevideos and captions for training our LDM.

## Video preparation

We focus on three key steps in preparing videos for training:

1. Filter inappropriate videos
2. 2.Standardize videos
3. 3.Precompute video representations in the latent space

## Filter inappropriate videos

Large datasets often contain unwanted content.This step removes inappropriate videos to ensure the model learns only from high-quality ones. Common steps include:

- ·Remove low-quality or short videos: We follow Movie Gen [4] and remove lowresolution, short, slow motion, or distorted videos with compression artifacts.
- Remove duplicated videos (deduplication): We use a deduplication method such as [7] to eliminate identical videos. This ensures training data is diverse and the model will notbe exposed to certain videosmore than others.
- ·Remove harmful videos:We use harm-detection models to identify and remove videos

346|Chapter 11.Text-to-VideoGeneration erateharmful videos.

## Standardizevideos

- data consists only of videos of the same length.
- to ensure all the videos have the same frame rates.
- 1280x720pixels.

## Precompute videorepresentations in the latent space

As theLDM operates in the latent space,it needs only latent representations as input. Thus,each trainingiterationnormallyrequires thefollowingsteps:

- 1.Extractframesfrom avideoin the trainingdata.
2. Pass these frames through a pretrained compression network to obtain latent representations.
3. Use the latent representations to continue training the diffusion model.

However, those steps are inefficient. Extracting frames and compressing them for millions of videos each time we train a new model slows diffusion training. Computing latent representations on the fly is resource-intensive and time-consuming.

<!-- image -->

Figure 11.9: Prepared video-caption data for training

<!-- image -->

| ID   | Videolatents   | Caption embeddings   |
|------|----------------|----------------------|
| N    |                |                      |

## Model Development

## Architecture

When selecting the architecture for a text-to-video diffusion model, we have two main options: U-Net and DiT. We'll examine each and determine the additional layers required to extend them to handle videos.

## U-Net for videos

by attending to the text prompt.

Figure11.10:U-Netarchitectureforimagegeneration

<!-- image -->

However, these layers mainly focus on capturing the relationships between pixels within a single image. This design presents a challenge for videos, where maintaining temporal consistency is crucial for smooth motion and continuity across frames. Current layers, however,operatespatiallywithinindividualframesratherthanacrossframes.

To address this shortcoming, we modify the U-Net architecture to understand relationships across frames.Inparticular,we inject two commonlyused temporal layers:

- ·Temporalattention
- ·Temporalconvolution

Figure 11.11: Injecting temporal layers into the U-Net's downsampling and upsampling blocks

<!-- image -->

350 |Chapter 11.Text-to-Video Generation Let's briefly review each layer.

theotherframes.

Figure 11.12: Temporal attention updating features by looking across frames

<!-- image -->

a 3D segment of data, to capture the temporal dimension. Figure 11.13 illustrates 2D and 3D temporal convolutions.

3D

Filter

3D convolution

2D

Filter

2D convolution

Figure 11.13:2D convolution vs.3D convolutions

In summary,to extend a U-Net architecture toprocessvideos,we can interleave temporal convolution and temporal attention layers in each downsampling and upsampling block. These layers allow the U-Net architecture to model the motion in input videos and generate a sequence of frames that are temporally consistent. To learn more about how these layers can be interleaved,refer to[10].

## DiT for videos

Unlike U-Net, which is based mainly on convolutions, DiT relies primarily on the Transformer architecture. As shown in Figure 11.14, DiT consists of four main components:

- ·Patchify

- Positional encoding
- ·Transformer
- ·Unpatchify

Figure11.14:DiT components

<!-- image -->

Let's examine each component and understand its purpose.

## Patchify

the input into smaller, fxed-size patches. Each patch is then flattened to form a sequence with the Transformer'shidden size.

input into fixed-size 2D patches. For videos, the video is divided into 3D patches.

352 |Chapter 11.Text-to-Video Generation

Figure11.15:Patchify forimage vs.video

<!-- image -->

## Positional encoding

The positional encoding component produces an embedding for each position in the original sequence. These embeddings provide the Transformer with information about the location of each patch in the original input.

Figure 11.16:1Dpositional encoding forimage vs.video

<!-- image -->

As we saw in Chapter 2, there are different ways to encode positions: Some methods use fixedpositional encoding during training,while other methods make the positional encoding learnable.There are also different ways to assign positions to each patch.For example,we cangive each patch a single number to show itsplace in a sequence or use 3D coordinates(2D forimages) to show where each patch is in space and time.

Figure 11.17: 1D, 2D, and 3D positional encoding

<!-- image -->

There is not one best way to do positional encoding. We often need to run experiments to find the approach that willbe most effective for the data and task. In this chapter, we follow OpenSora [11] and use RoPE [12] positional encoding. To learn more about

354|Chapter11.Text-to-VideoGeneration

<!-- image -->

<!-- image -->

## Training

training are:

- 1.Noise addition:A timestep is randomly sampled to determine the level of noise addition.Thesampled timestepisused to addnoise to theinputvideo.
- 2.Noise prediction:The DiT model receives the noisy video asinput and predicts the added noisebased onconditioning signals suchas text prompt and the sampled timestep.
- 3.Loss calculation:The loss is measured by comparing thepredicted noise to the actual noise.

To review diffusion training inmore detail,refer to Chapter 9.

## ML objective and lossfunction

The primary loss function is the reconstruction loss,calculated using the mean squared error (MsE) formula.This loss measures the difference between the predicted noise and the actual noise, encouraging the model to accurately predict the added noise. The ML objective is to minimize the reconstruction loss,leading to accurate video reconstruction.

Researchers have experimented with utilizing other loss functions to enhance text-tovideo performance. To learn more, refer to [4].

## Challenges in training video diffusion models

Training a DiT model for text-to-video generation involves several challenges and design decisions. This section explores two important challenges:

- ·Lack of large-scale video-text data
- ·Computational cost of high-resolution video generation

## Lack of large-scale video-text training data

- textdata,

Model Development |357

The pretrained model is then finetuned on video-text pairs for video generation.

Figure 11.20:Two strategies to utilize image-text training data

<!-- image -->

Both strategies leverage hundreds of millions of image-text data in training, allowing the DiT model to learn from both images andvideos.For simplicity,we choose thefirststrategy,as itrequiresonly one stage of training.However,both strategiescan be effective in practice.

## Computational costof high-resolution video generation

As discussed earlier, processing and generating videos is more expensive than images. This isprimarily because videos generally contain hundreds of frames,making the process slower andmorecostly.Generatinghigh-resolutionvideos,such as720por 1080p,adds to the challenge.

Here are a few common strategies to reduce the computational cost of training highresolutionvideogeneration models:

- ·Employ an LDM-based approach: Instead of training the DiT model directly in pixel space, we use a compression network to convert videos from pixel space into a lowerdimensional latent space. Training the difusion model in this latent space reduces the computational load,
- tent space before training,we avoid repetitive computations during training.Utilizing this cached data speeds up the trainingprocess.

358|Chapter 11.Text-to-Video Generation to 1080p or4K.

- interpolatetoachieve24FPs.
- ·Use more efficient architectures: We can adopt an efficient implementation of the tionally, techniques like Mixture of Experts (MoE) [17] can be used to accelerate the trainingprocess.
- ·Use distributed training: We use distributed training techniques, such as tensor parallelism,toparallelize training across multiple devices.By splitting the model,data,or both acrossdifferent devices,we can significantly speed up trainingand handlelarger video datasets more efficiently.This approachisparticularlyuseful forhigh-resolution video generation,where memory and computational demands are substantial.Foran overview of distributed training,refer to Chapter 1.

Figure 11.21:Efficient text-to-video pipeline

<!-- image -->

## Sampling

<!-- image -->

<!-- image -->

sponding text.

## Human evaluation

measures.

For human evaluation,we generate videos from test prompts using two different models. consistency. This process allows us to compare two models to see which one performs better.

## Online evaluation metrics

Online evaluation metrics for text-to-video models are similar to those for text-to-image models.Importantmetricsinclude:

- Click-throughrate
- ·Time spent on the page
- Userfeedback
- ·Conversionrate

These metricshelp gauge user engagement,satisfaction,and overallmodel performance in production.

## Overall ML System Design

In this section,we dive into the holistic design of a text-to-video generation system. In particular,we examine the following pipelines:

- ·Data pipeline
- ·Training pipeline
- ·Inference pipeline

## Data pipeline

pute and store caption embeddings.

Figure 11.23:Datapipeline

<!-- image -->

## Training pipeline

The training pipeline trains the model using the training data prepared by the data pipeline.

## Inference pipeline

The inference pipeline processes real-time user requests to generate videos from text prompts. As shown in Figure 11.24,it has several crucial components that ensure systemqualityandsafety.

Prompt

Auto-Complete

Type

prompt Submit

②→

prompt Prompt Safety

Service

Safe?

Reject Prompt Enhancer

4

Video

Generator

Text

(Encoder)

LDM

Visual

Decoder Harm Detector Safe?

Super-Resolution

Ba-

No Yes

Reject

request

Figure 11.24:Inference pipeline components

Overall ML SystemDesign|363

Temporal

Spatial

Super-Resolution Generated video

- ·Visual decoder
- ·Temporal super-resolution

## Visual decoder

pixel space.

## Temporalsuper-resolution

This component interpolates between the generated frames, leading to smoother motion invideos.

## Other TalkingPoints

In case there's extra time at the end of the interview,youmight discuss these further topics:

- Ensuring samplingflexibilityforvariable durations,resolutions,and aspectratios[1].
- ·Extending the text-to-video model to downstream applications such as inpainting, outputpainting,video-to-videostylization,frameinterpolation,super-resolution,and animating images(image-to-video)[10].
- Supportforcontrolling thegeneratedvideossuch as the level of desired motion and the type of motion (camera vs.object motion)[27].
- Using progressive distillation techniques to reduce the computational demands of training[28].
- Details of spatial and temporal super-resolutionmodels[15].
- Details of re-captioning model [9,8].
- Different noise schedulers[29].
- ·Noise conditioning augmentation techniques[30].
- Personalizinga text-to-video model toaparticular subject[31].
- ControlNetfor text-to-videomodels[32].
- DetailsofStableCascademethod[33].
- Detailsofvisual compressionnetwork[13].

<!-- image -->