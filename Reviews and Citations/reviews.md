## BMVC 2020 Reviews

### Reviewer 1

1. [ Paper Summary ] What are the key ideas, what is the significance and how are the ideas validated? Please be concise (3-5 sentences).
This paper presents a new activation function (Mish) which was found empirically by experimenting with functions that behave similarly to Swish. Mish was extensively tested on various tasks, datasets, and network architectures, and generally provided better performance and improved convergence and stability of training than several compared functions. An efficient GPU implementation is also tested.
2. [ Paper Strengths ] Please summarise the strengths of the paper. (Eg: novelty, insight, theoretical strength, state of the art performance, thorough evaluation). Please provide a clear explanation of why they are valuable.
Mish is a new activation function, although by design with similar properties to previous ones. It outperforms other compared functions on several scenarios. The evaluation is reasonably thorough, although a few comparisons against the state-of-the-art Swish are missing.
3. [ Paper Weaknesses ] Please summarize the weaknesses of the paper. (E.g., Lack of novelty, technical errors, insufficient evaluation, etc). You should clearly justify your criticisms with precise and factual comments (E.g., with an explanation of technical errors, citation to prior work if novelty is an issue). Please note: It is not appropriate to ask for comparison with unpublished arXiv papers, and papers published after the BMVC deadline. Please be polite and constructive.
The main results are a generally better performance in deeper networks, against input noise, and for a variety of tasks, architectures, and weight initialisation methods. It would be very interesting to have some more in-depth discussion of these results and of the possible reasons for them. The current discussion stresses a lot the similarities with Swish. On the other hand, the differences with Swish that may explain their different results are not discussed. For example, I notice in Fig. 1b that the second derivative of Mish is much more symmetrical. What could be the effect of this property? Could this explain some of the results such as better convergence on large networks??

I am a little bit concerned by the comment "a threshold of 20 is enforced on Softplus which makes the training more stable and prevents gradient overflow". How does this threshold modify the curves provided in Fig. 1 and the discussions on the properties of Mish and of its smooth gradient (Fig. 3)?

Section 4.3: the authors state that they use the validation set to compare methods. Is this a typo? If no, why not using the test set?

5. [ Justification of Rating ] Please explain how the different strengths and weakness were weighed together to produce the overall rating. Please provide details of questions or ambiguities to be addressed in the rebuttal that might change your rating.
The new proposed function obtains interesting results, but I am concerned by the limited discussions, that may further be misled by the modified implementation mentioned in point 3.
6. [ Suggestions to Authors ] Any further comments to the authors and suggestions for improving the manuscript (e.g. typos). These are not relevant for your rating.
Since Swish is the main function to compare against, it would be nice to complete tables 3 and 4 with Swish results, so as to compare more clearly the respective performances of the functions in networks of different complexities and depths.

Similarly, Fig. 4a stops at 25 layers, while some experiments use the larger ResNe(x)t-50 architectures. Therefore it would be interesting to continue this plot up to 50 layers.

Could the authors provide some indications on how the Cuda implementation is obtained?

### Reviewer 2:

1. [ Paper Summary ] What are the key ideas, what is the significance and how are the ideas validated? Please be concise (3-5 sentences).
This paper proposed a novel activation function called Mish. Experiments on image classification and object detection validated that marginal improvement can be achieved.
2. [ Paper Strengths ] Please summarise the strengths of the paper. (Eg: novelty, insight, theoretical strength, state of the art performance, thorough evaluation). Please provide a clear explanation of why they are valuable.
This paper proposed a novel activation function that can be adopted in various network architectures to acquire performance gains.
3. [ Paper Weaknesses ] Please summarize the weaknesses of the paper. (E.g., Lack of novelty, technical errors, insufficient evaluation, etc). You should clearly justify your criticisms with precise and factual comments (E.g., with an explanation of technical errors, citation to prior work if novelty is an issue). Please note: It is not appropriate to ask for comparison with unpublished arXiv papers, and papers published after the BMVC deadline. Please be polite and constructive.
  a. The insight of the proposed activation function is not enough.
  b. In table 1, the mean loss and standard deviation of accuracy of Mish are NOT the lowest, they are mistakenly marked bold in the table and described in the text.
  c. The computational complexity of Mish is much higher than RELU, however, the performance improvement is marginal (e.g. Table 4)

5. [ Justification of Rating ] Please explain how the different strengths and weakness were weighed together to produce the overall rating. Please provide details of questions or ambiguities to be addressed in the rebuttal that might change your rating.
  a. The high computational complexity and the marginal performance gain may hinder the application of the proposed activation function in the community.
  b. The insight is not enough.
  c. Some basic mistakes exist in Table 1.
6. [ Suggestions to Authors ] Any further comments to the authors and suggestions for improving the manuscript (e.g. typos). These are not relevant for your rating.
  a. give more theoretical insight
  b. For ImageNet classification, these original mainstream network architectures should be used as baselines to better justify the contribution of the proposed activation function.
  c. The contribution of data augmentation to performance gain is not quite relative to this work and should be shortened or omitted in the text.

### Reviewer 3:

1. [ Paper Summary ] What are the key ideas, what is the significance and how are the ideas validated? Please be concise (3-5 sentences).
This paper introduces a self-regularized non-monotonic activation function, Mish. Mathematically, Mish multiplies the non-modulated input with the output of a non-linear function of the input so that a small amount of negative information is preserved. Mish has two properties, unbounded above and bounded below. These properties lead to higher performance for classification and object detection compared to other activation functions. The authors conduct extensive experiments on different datasets for different computer vision tasks using different types of deep networks in terms of performance and training stability.
2. [ Paper Strengths ] Please summarise the strengths of the paper. (Eg: novelty, insight, theoretical strength, state of the art performance, thorough evaluation). Please provide a clear explanation of why they are valuable.
The authors perform extensive experiments to depict the effectiveness of Mish, including different network structures, different benchmarks, and different computer vision tasks. The authors also explore some strategies which might affect the proposed activation functions, including the depth of the neural networks, input Gaussian noise, and the weight initialization methods.
3. [ Paper Weaknesses ] Please summarize the weaknesses of the paper. (E.g., Lack of novelty, technical errors, insufficient evaluation, etc). You should clearly justify your criticisms with precise and factual comments (E.g., with an explanation of technical errors, citation to prior work if novelty is an issue). Please note: It is not appropriate to ask for comparison with unpublished arXiv papers, and papers published after the BMVC deadline. Please be polite and constructive.
The paper seems to be experimental so that the theoretical analysis is relatively weak. The authors did experiments for demonstration in terms of accuracy of image classification and object detection. For computational complexity (e.g. Eq, (4) for the backward pass), the analysis is relatively weak as well.

Moreover, the writing needs to be improved further or polished, such as Tense utilization.

5. [ Justification of Rating ] Please explain how the different strengths and weakness were weighed together to produce the overall rating. Please provide details of questions or ambiguities to be addressed in the rebuttal that might change your rating.
-- To be a general solution needs be more validation. Take the experiments in Figure 4 as an example, the results in Figure 4(a)-(c) were collected with using fully-connected networks, a five-layered convolutional network, and a six-layered convolutional network, respectively. Will these three activation functions keep the same trends with the reported results in Figure 4 if the functions are tested on the different nets? e.g. the same experiment from Figure 4(a) is done using a five-layered convolutional network instead of fully-connected layers.

-- Could the authors make some training stability analysis with the function g(x) = tanh(softplus(x)) together with arctan(x)softplus(x) and tanh(x)softplus(x) in Section 2?

-- The activation function formulation in Eq. (3). i.e. f(x) = x tanh(softplus(x)). As we can see that the activation function is related to the non-modulated input x, which is something like the skip connection in the Residual Network (adding the activation F(x) with the original x). Therefore, the activation process might be highly related to the distributions of the input x. The authors should make more explorations or theoretical analysis of this. For instance, how the introduced activation function will influence the network training, performance boost, or training stability if the input x is normalized or non-normalized? At least, the authors need to perform some experiments on the ablation study for demonstrations.

-- Section 4.5 seems to be unclear for me, e.g. the network structure, the datasets etc. On the one hand, we can find that the 1st derivative of Mish in Eq. (4) are complex obviously, and might be complicated than that of the softplus function. Thus the back-propagation process is higher computational complexity for the Eq. (4). However, the results in Table 5 seems to be unsupportive. For Datatype fp 16, the backward pass time for softplus is 488.5, and higher than 345.6 of Mish-CUDA. For Datatype fp 32, there is another different trend. Could the author give some more explanations? One the other hand, it is unclear how to make a trade-off among stability, accuracy, and efficiency. Meanwhile, considering the complexity of Eq. (4), will the activation function be feasible for some deeper networks, like ResNet-152?

-- Are there some data missing in Table 3 and 4?
6. [ Suggestions to Authors ] Any further comments to the authors and suggestions for improving the manuscript (e.g. typos). These are not relevant for your rating.
Line 81.5 and Line 111: it’s ---> its
Line 158: twice a
Line 211: arbitrarily ---> arbitrary

### Reviewer 4: 

1. [ Paper Summary ] What are the key ideas, what is the significance and how are the ideas validated? Please be concise (3-5 sentences).
This paper proposes a new activation function, and provide mathematical insight for it. Some basic experiments like landscape, depth-by-depth accuracy, robustness to input noise, weight initialization is explored. They can collect consistent gain on ImageNet classification, MS-COCO detection after deploying the proposed activation function.
2. [ Paper Strengths ] Please summarise the strengths of the paper. (Eg: novelty, insight, theoretical strength, state of the art performance, thorough evaluation). Please provide a clear explanation of why they are valuable.
- Well written and easy to follow
- Extensive experiments to explore this kind of newly proposed activation function.
- Consistent gain in the task of classification and detection.
3. [ Paper Weaknesses ] Please summarize the weaknesses of the paper. (E.g., Lack of novelty, technical errors, insufficient evaluation, etc). You should clearly justify your criticisms with precise and factual comments (E.g., with an explanation of technical errors, citation to prior work if novelty is an issue). Please note: It is not appropriate to ask for comparison with unpublished arXiv papers, and papers published after the BMVC deadline. Please be polite and constructive.
- The numbers in table 2,3,4 are all trained by yourself under the fair comparison setting(same codebase), say it explicitly if so.

5. [ Justification of Rating ] Please explain how the different strengths and weakness were weighed together to produce the overall rating. Please provide details of questions or ambiguities to be addressed in the rebuttal that might change your rating.
I am inclined to give borderline accept this time, I may change my score if my concern is resolved.
6. [ Suggestions to Authors ] Any further comments to the authors and suggestions for improving the manuscript (e.g. typos). These are not relevant for your rating.
Code releasing will facilitate the follow-ups
