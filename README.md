# The MisinfoPy Model

## Authored by Felicitas Reddel

The goal of this project was to create an extendable agent-based model of misinformation spread on social media and use it to explore structural uncertainty in the way of how agents update their beliefs. The MisinfoPy model has been implemented in Python (3.9) and is based on the agent-based libray [Mesa](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj8turvu875AhU0xAIHHVZwC-8QFnoECAUQAQ&url=https%3A%2F%2Fmesa.readthedocs.io%2F&usg=AOvVaw1pHefbRMgtD1z_nfEh3p8y). To explore uncertainties, we make extensive use of the [ema_workbench](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwivl8r-u875AhVBxQIHHQ8LAVwQFnoECAcQAQ&url=https%3A%2F%2Femaworkbench.readthedocs.io%2F&usg=AOvVaw1G-T7qhwF2bOUvDgsMZKrF). This project has been developed within a Master's Thesis at the TU Delft during the academic year 2021/22 between quarter 3 and 4. A summary of this research project can be found below.

---

## Summary

Misinformation on social media is an urgent grand challenge. Misinformation has caused excess deaths because people abstain from getting vaccinated and other evidence-based prevention behaviors. Misinformation also influences various other important topics such as climate change. And it has the potential to influence countless other areas. In order to exemplify the broader point, this project focuses on simple example of beliefs around the safety of COVID-19 vaccinations. To find ways to successfully tackle this grand challenge, it is crucial to have thorough understanding of the system.
Modeling and simulation can be a powerful tool to support our reasoning about big and complex systems such as misinformation on social media. Therefore, I choose to look at how modeling and simulation can be useful for the study of misinformation on social media and of potential counter-measures.

Agent-based models (ABMs) are one of the useful modeling paradigms for this grand challenge. And while there is a body of literature on ABMs in the field of misinformation research, there is structural uncertainty about how to represent the way that people change their minds on social media. Different types of representations of this updating process are used. It is unclear which of them is the most suitable representation of the real-world process and also to what extent it makes a difference for the choice of counter-measures. Moreover, the choice between these different belief update functions is usually not discussed. And to the best of my knowledge, nobody has explored the issue of whether the choice between belief update functions makes a substantial difference in the conclusions from the studies. 

Because of the significance of this grand challenge and the lack of exploring a key structural uncertainty, I choose to apply a method for exploring uncertainty in the context of ABMs. More specifically, because the structural uncertainty about the belief update function is a central component of models in this field, I explore a method for handling this structural uncertainty. This project is a show case of the value that methods from the field of Decision-Making Under Deep Uncertainty (DMDU) have for the field of misinformation focused ABMs.

Yet, applying a DMDU approach is not only useful for enabling exploration of uncertainties. With many DMDU methods, it is possible to evaluate policies based on not only a single, but on multiple objectives. As far as I know, also the evaluation of multiple objectives has not previously been done in the field of opinion dynamic models such as ABMs which focus on misinformation on social media. However, policies that aim at tackling the misinformation challenge do not only impact one single stakeholder, but a multitude of diverse stakeholders who care about various aspects of the system. If we pick policies by only optimizing for one objective, we run the risk of merely shifting the problem. To find solutions that are sustainable and work for the whole system, it is helpful to consider multiple metrics that stakeholders care about. 
The ranking and filtering by multiple objectives is not trivial. But there is a method called non-dominated ranking which can be applied to do exactly that. This results in so-called Pareto-optimal policies. It is in this specific niche that I pursue the following methodological question within the field of agent-based misinformation modeling:

---
### Main Research Question

How does the consideration of structural uncertainty with respect to the choice between different belief update functions influence the resulting Pareto-optimal policies and their performance?

---

I look at three alternative belief update functions, where each belief update function is represented by one model. I show that the choice of the belief update function makes a significant difference for what kind of policies are Pareto-optimal and for the outcomes that stem from these policies. To investigate how the choice of the belief update function influences which policies are Pareto-optimal and what kind of outcomes result, I apply the DMDU-method of Many-Objective Robust Decision-Making (MORDM) approach. With DMDU methods, modellers can acknowledge the uncomfortable situation in which we know that we have uncertainties, ruining the possibility of using models as reliable prediction machines. These uncertainties can be about the real world's states (i.e., parametric uncertainties) or its processes (i.e., structural uncertainty). When applying DMDU methods, modellers can aim to find policies that perform robustly over a large number of possible instantiations of parametric or structural uncertainties. In this project, I first evaluate more than 26'000 candidate policies with each of the three belief update functions. Then, I select a set of Pareto-optimal policies for each belief update function. Additionally, I select a set of policies that seem optimal when only considering a single metric.  Subsequently, I re-evaluate Pareto-optimal policies of each belief update function under deep uncertainty to gain a better impression of their performance. Finally, I compare the commonalities and differences between the selected policies and their performances. This, I do for either method of selection and for all three belief update functions.

To explore the structural uncertainty, I use a model which can be instantiated with either of the alternative belief update functions. I refer to these three possible instantiations as the three different models. The first model uses the commonly used function based on the research by Deffuant (hereafter 'DEFFUANT model'). In it, beliefs are always updated by a fix percentage towards the newly incoming information. In this project, this newly incoming information is the belief that is represented in a seen post. The second model samples whether a belief update happens or not. If an update happens, the new belief is the average between the previous belief and the newly incoming information. We call this the 'SAMPLE model'. Unfortunately, neither of these two models includes well-established phenomena from social psychology. Examples of such phenomena include for instance that we are more willing to update towards beliefs that are more similar to ours, that we have limited attention capacity, and that it takes more to change someone's mind when they are very convinced of their current belief than when they are uncertain. The third model was chosen to fill this void by basing its belief update function on Social Impact Theory (SIT) and adjusting this theory to the context of social media. This model is referred to as the 'SIT model'. 

---
### Main Findings

- There is a clear distinction between the models' optimal policies as well as their outcomes.
- Differences in parameters do make a difference.
- The models' optimal policies exhibit an order in how optimistic their outcomes are. This order (in descending direction) is DEFFUANT, SAMPLE, and SIT.
- The outcomes of the DEFFUANT and the SAMPLE model are more similar to each other than to the SIT model.
---

The main methodological take-away is that the DMDU approach can bring substantial value to the field of ABM-based studies on the grand challenge of misinformation on social media platforms. While this is shown by a simple exploration of the structural uncertainty with respect to the belief update, many more insights could be gathered by utilizing the DMDU approach. For instance, the DMDU approach offers state-of-the-art methods to identify vulnerable scenarios, i.e., scenarios which would be particularly bleak. Another example could be to explore different problem formulations with different sets of objectives or other structural uncertainties such as the posting behavior. 

Furthermore, by utilizing the tools of DMDU, also society as a whole can benefit. By including multiple objectives and a wide range of considered uncertainties, the many different world-views and values of the diverse stakeholders can be taken into account in order to avoid potential policy gridlock situations. This could contribute to tackling the misinformation grand challenge more successfully and thus for instance lead to more people embracing evidence-based medical interventions.
