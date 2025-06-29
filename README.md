# Credit Risk Model for Bati Bank

This repository contains the end-to-end implementation for building, deploying, and automating a credit risk model for Bati Bank's new Buy-Now-Pay-Later (BNPL) service.

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord fundamentally shapes our project by establishing strict standards for how financial institutions measure and manage risk. The Accord's Internal Ratings-Based (IRB) approach allows banks like Bati Bank to use their own internal models to calculate credit risk and determine capital requirements. However, this privilege comes with significant regulatory oversight.

This directly leads to the following needs for our model:

*   **Interpretability:** Regulators, auditors, and internal risk managers must be able to understand *why* the model assigns a certain risk probability to a customer. A "black box" model, regardless of its accuracy, is unacceptable because its decision-making process is opaque. We must be able to explain the specific factors that contribute to a customer's score, ensuring the model is fair, logical, and non-discriminatory.

*   **Documentation:** Every stage of the model development lifecycle—from data selection and cleaning to feature engineering choices and model validation—must be meticulously documented. This creates a clear audit trail, proving that the model is built on sound statistical principles and its performance is well-understood.

*   **Conceptual Soundness:** The model must be robust and its predictions stable over different economic conditions. An interpretable model allows us to validate that the relationships it has learned are logical from a business perspective, rather than being statistical artifacts of the training data.

In short, Basel II forces us to prioritize transparency and rigor over pure predictive power, making model interpretability a non-negotiable requirement.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

**Necessity of a Proxy Variable:**
A supervised machine learning model learns by finding patterns that connect input features (e.g., customer transaction history) to a known outcome label (the target variable). Our e-commerce dataset does not contain a `has_defaulted` column, as customers have not yet been issued loans. To overcome this "cold start" problem, we must engineer a **proxy variable**.

We are creating this proxy by identifying behaviors that we hypothesize are correlated with higher credit risk. In this project, we define "high-risk" customers as those who are highly disengaged from the e-commerce platform (e.g., high recency, low frequency, low monetary value). This behavioral proxy allows us to train a model that can differentiate between customer profiles *before* any real default data is available.

**Potential Business Risks:**
The primary risk is **proxy-target misalignment**, where our proxy for risk (disengagement) does not accurately reflect a customer's true willingness or ability to repay a loan. This can lead to two critical errors:

*   **False Positives (Incorrectly Flagging Good Customers):** The model might classify a loyal, low-risk customer as "high-risk" because their purchasing pattern fits the proxy definition (e.g., a customer who makes one large, valuable purchase per year). The business impact is significant: we would deny credit to a valuable customer, leading to lost revenue and customer dissatisfaction.

*   **False Negatives (Failing to Flag Bad Customers):** The model might classify a genuinely high-risk individual as "low-risk" because their behavior does not match the disengagement proxy (e.g., a new user who makes several fraudulent or high-velocity purchases with no intention of repaying). This would lead to approving bad loans, resulting in direct financial losses for Bati Bank.

Therefore, while necessary, the proxy-based model must be continuously monitored and validated against actual loan performance data as it becomes available.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice between a simple and a complex model represents a critical trade-off between **interpretability** and **predictive performance**.

| Feature                | Simple Model (Logistic Regression + WoE)                                     | Complex Model (Gradient Boosting)                                       |
| ---------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Interpretability**   | **High.** Each feature's contribution to the final score is clear, direct, and quantifiable. This is ideal for regulatory review and explaining decisions. | **Low.** It is a "black box" by nature. While methods like SHAP can provide feature importance, they don't offer the same level of granular, transparent logic. |
| **Performance**        | **Good, but often lower.** It assumes linear relationships and may miss complex, non-linear interactions between features that a more powerful model could capture. | **High.** Typically achieves superior predictive accuracy (AUC) by modeling intricate patterns and feature interactions in the data. |
| **Regulatory Approval**| **Easier to achieve.** Its transparency and long history as an industry standard make it familiar and acceptable to regulators. | **More challenging.** Requires extensive justification, validation, and explainability analysis (e.g., SHAP reports) to convince regulators of its stability and fairness. |
| **Risk of Overfitting**| **Low.** Simpler models are less prone to memorizing noise in the training data. | **Higher.** Requires careful and extensive hyperparameter tuning to prevent overfitting and ensure it generalizes well to new data. |

**Strategic Choice for Bati Bank:**
For our initial launch, a **Logistic Regression model with Weight of Evidence (WoE) transformation** is the most prudent choice. It provides a strong, defensible, and interpretable baseline that aligns with the stringent requirements of a regulated financial context. A **Gradient Boosting model** should be developed as a "challenger" model. If it demonstrates a significant and stable performance lift, it can be considered for a future deployment, armed with a robust framework for explaining its predictions.