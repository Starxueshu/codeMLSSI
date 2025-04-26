#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)


# In[2]:


import numba


# In[3]:


numba.__version__


# In[4]:


from numba.np.ufunc import _internal


# In[5]:


import sklearn


# In[6]:


sklearn.__version__


# In[7]:


#conda install scikit-learn


# In[8]:


#conda install scikit-learn-intelex
#pip install scipy


# In[9]:


#from sklearn.base import BaseEstimator, TransformerMixin


# In[10]:


#import pandas as pd


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 8, 8
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率


# import dataset

# In[12]:


star = pd.read_csv(r'F:/materi/12-SXD/postoperative infection/数据集2011-2020-265例-SXD返回2.0 -增加数据集.csv')
#data_train = pd.read_csv(r'F:/materi/12-SXD/postoperative infection/start0.7.csv')
#data_test = pd.read_csv(r'F:/materi/12-SXD/postoperative infection/starv0.3.csv')


# In[14]:


X_res = star.loc[:,[
"WBC","Glucose",
"Tumortype",
"Primarytumor",
"Age",
"Gender",
"Smoking",
"BMI",
"Numberofcommorbidity",
"Coronarydisease",
"Diabetes",
"Hypertension",

"Preoperativechemotherapy",
"Preoperativetargeted",
"Preoperativeendocrinology",
"Extravertebralbonemetastasis",
"Viscerametastases",
"ECOG",
"Preoperativeembolization",
"Surgicaltime",
"Surgicalprocess",
"Bloodtransfusion",

"Surgicalsite",
"Surgicalsegements",
"Albumin",

"HGB",
"PLT"
]]


# In[15]:


y_res = star.SSI


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=0)


# In[17]:


y_test.to_excel('F:/materi/12-SXD/postoperative infection/y_test.xlsx')


# In[18]:


X_test.to_excel('F:/materi/12-SXD/postoperative infection/X_test.xlsx')


# In[20]:


X_train.to_excel('F:/materi/12-SXD/postoperative infection/X_train.xlsx')


# In[19]:


y_train.to_excel('F:/materi/12-SXD/postoperative infection/y_train.xlsx')


# In[17]:


from sklearn import datasets


# In[18]:


star.Age.shape


# In[19]:


star.data = star.loc[:,[
"Surgicaltime",
"Tumortype",
"Smoking",
"Numberofcommorbidity",
"Diabetes",
"Surgicalsegements","Viscerametastases"
]]


# In[20]:


from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_res, y_res = smote_tomek.fit_resample(star.data, star.SSI)


# In[28]:


X_res.to_excel('F:/materi/12-SXD/postoperative infection/X_res.xlsx')


# In[29]:


y_res.to_excel('F:/materi/12-SXD/postoperative infection/y_res.xlsx')


# In[23]:


X_res.Diabetes.value_counts()


# In[24]:


y_res.value_counts()


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=0)


# In[32]:


X_train.shape, y_train.shape


# In[33]:


pd.set_option('display.max_columns', X_train.shape[1])
pd.set_option('max_colwidth', 1000)


# In[34]:


X_train.head()


# In[35]:


#data_train.info()


# Data preprocessing piprlines
# Prepare the data to a format that can be fit into scikit learn algorithms

# Categorical variable encoder

# In[37]:


categorical_vars = ["Tumortype",
"Smoking",
"Numberofcommorbidity",
"Diabetes",
"Surgicalsegements","Viscerametastases"
]


# In[38]:


X_train[categorical_vars].head()


# In[39]:


# to make a custom transformer to fit into a pipeline
class Vars_selector(BaseEstimator, TransformerMixin):
    '''Returns a subset of variables in a dataframe'''
    def __init__(self, var_names):
        '''var_names is a list of categorical variables names'''
        self.var_names = var_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''returns a dataframe with selected variables'''
        return X[self.var_names]


# In[40]:


class Cat_vars_encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a dataframe'''

        return X.values


# Transform data in a pipeline

# In[44]:


# categorical variables preprocessing
cat_vars_pipeline = Pipeline([
    ('selector', Vars_selector(categorical_vars)),
    ('encoder', Cat_vars_encoder())
])


# For many machine learning algorithms, gradient descent is the preferred or even the only optimization method to learn the model parameters. Gradient descent is highly sensitive to feature scaling.

# ** Continuous vars **

# In[45]:


continuous_vars = ["Surgicaltime"]


# In[46]:


X_train[continuous_vars].describe()


# In[51]:


X_train[continuous_vars].head()


# In[47]:


# continuous variables preprocessing
cont_vars_pipeline = Pipeline([
    ('selector', Vars_selector(continuous_vars)),
    ('standardizer', StandardScaler())
])


# In[48]:


preproc_pipeline = FeatureUnion(transformer_list=[
    ('cat_pipeline', cat_vars_pipeline),
    ('cont_pipeline', cont_vars_pipeline)
])


# In[49]:


data_train_X = pd.DataFrame(preproc_pipeline.fit_transform(X_train), 
                            columns= categorical_vars +continuous_vars)


# In[50]:


data_train_X.head()


# Fitting classifiers

# In[52]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import learning_curve, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


# In[53]:


y_train.value_counts()


# In[54]:


y_train


# This is a fairly balanced dataset(i.e., the number of positive and negative cases are roughly the same), and we'll use AUC as our metric to optimise the model performance.

# Assessing learning curve using the model default settings
# Tuning the model hyper-parameters are always difficult, so a good starting point is to see how the Scikit-learn default settings for the model performs, i.e., to see if it overfits or underfits, or is just right. This will give a good indication as to the direction of tuning.

# In[55]:


def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, scoring='roc_auc',
                                                           random_state=42, n_jobs=-1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="training scores")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), "o-", label="x-val scores")
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.xlabel("Training set size")
    plt.ylabel("Area under Curve")
    plt.title('{} learning curve'.format(model.__class__.__name__))


# # Compute and compare test metrics
# Transform test data set

# In[60]:


data_test_X = pd.DataFrame(preproc_pipeline.transform(X_test), # it's imperative not to do fit_transfomr again
                           columns=categorical_vars + continuous_vars)


# In[61]:


data_test_X.shape


# In[62]:


y_test.value_counts()


# In[63]:


data_test_X.head()


# In[64]:


def plot_roc_curve(fpr, tpr, auc, model=None):
    if model == None:
        title = None
    elif isinstance(model, str):
        title = model
    else:
        title = model.__class__.__name__
#    title = None if model == None else model.__class__.__name__
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label='auc: {}'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-.01, 1.01, -.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(fontsize=14).get_frame().set_facecolor('white')
    plt.title('{} - ROC Curve'.format(title))


# # Logistic Regression---网格搜索模型调参GridSearchCV

# In[65]:


lr_clf = LogisticRegression(n_jobs = -1)
plot_learning_curves(lr_clf, data_train_X, y_train)


# Let's see if we can squeeze some more performance out by optimising C

# In[66]:


param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
lr_clf = LogisticRegression(random_state=42)
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, return_train_score=True,
                                cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(data_train_X, y_train)


# In[67]:


cv_rlt = grid_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# # Looks like C=10 is our best value.

# 下方with open,"wb",确实是保存了新的pkl模型文件，下一次使用该文件只需要直接调用即可

# In[69]:


lr_clf = grid_search.best_estimator_
with open('F:/materi/12-SXD/postoperative infection/lr_clf_final_round.pkl', 'wb') as f:
    pickle.dump(lr_clf, f)


# In[70]:


plot_learning_curves(lr_clf, data_train_X, y_train)


# Looks like the logistic regression model would benefit from additional data.

# # Logistic Regression model-ROC

# In[71]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/lr_clf_final_round.pkl', 'rb') as f:
    lr_clf = pickle.load(f)
lr_clf.fit(data_train_X, y_train)


# Accuracy scores

# In[72]:


accu_lr = accuracy_score(y_test, lr_clf.predict(data_test_X))


# In[73]:


round(accu_lr,3)


# In[74]:


pd.crosstab(y_test, lr_clf.predict(data_test_X))


# In[75]:


pred_proba_lr = lr_clf.predict_proba(data_test_X)


# In[76]:


#lr_clf.predict(data_test_X)


# In[77]:


y_test


# In[78]:


fpr, tpr, _ = roc_curve(y_test, pred_proba_lr[:, 1])
auc_lr = roc_auc_score(y_test, pred_proba_lr[:, 1])


# In[79]:


lr_score = lr_clf.score(data_test_X, y_test)
lr_accuracy_score=accuracy_score(y_test,lr_clf.predict(data_test_X))
lr_preci_score=precision_score(y_test,lr_clf.predict(data_test_X))
lr_recall_score=recall_score(y_test,lr_clf.predict(data_test_X))
lr_f1_score=f1_score(y_test,lr_clf.predict(data_test_X))
lr_auc=roc_auc_score(y_test,pred_proba_lr[:, 1])
print('lr_accuracy_score: %f,lr_preci_score: %f,lr_recall_score: %f,lr_f1_score: %f,lr_auc: %f'
      %(lr_accuracy_score,lr_preci_score,lr_recall_score,lr_f1_score,lr_auc))


# In[80]:


round(lr_auc,3)


# In[ ]:





# In[81]:


brier_score_loss(y_test, pred_proba_lr[:, 1])


# In[82]:


print('loss lr:', log_loss(y_test, pred_proba_lr[:, 1]))


# In[83]:


lr_results = [lr_accuracy_score,lr_preci_score,lr_recall_score,lr_f1_score,lr_auc,brier_score_loss(y_test, pred_proba_lr[:, 1]),log_loss(y_test, pred_proba_lr[:, 1])]


# In[84]:


lr_results = pd.DataFrame(lr_results)


# In[85]:


lr_results.columns = ['LR']


# In[86]:


lr_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]


# In[87]:


lr_results


# In[88]:


lr_results.to_excel('F:/materi/12-SXD/postoperative infection/lr_results.xlsx')


# In[ ]:





# In[89]:


round(auc_lr,3)


# In[90]:


plot_roc_curve(fpr, tpr, round(auc_lr,3), lr_clf)


# In[94]:


X_test['lr_pred_proba'] = pred_proba_lr[:, 1]


# In[95]:


#X_test.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-lr.csv'.format(len(X_test)), index=False)


# In[96]:


lr_his = X_test.join(y_test)


# In[97]:


lr_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-lr.csv'.format(len(lr_his)), index=False)


# In[98]:


lr_his.shape


# In[99]:


['#00468BFF','#ED0000FF','#42B540FF','#0099B4FF','#925E9FFF','#FDAF91FF','#AD002AFF','#1B1919FF']


# In[103]:


def plot_class_breakdown_hist(df, var, var_name, plot_title, xlog=False, ylog=False, **histkwargs):
    df[var][df.Hospitalinfection == 0].hist(alpha=.5, label='Negative', color = "#00468BFF", **histkwargs)
    df[var][df.Hospitalinfection == 1].hist(alpha=.5, label='Positive', color = "#ED0000FF", **histkwargs)
    plt.xlabel(var_name)
    plt.title(plot_title)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.ylim(ymax=600, ymin=0)
    plt.legend()
    plt.savefig(var_name + ' Class Breakdown.png');


# In[104]:


#plot_class_breakdown_hist(lr_his, 'lr_pred_proba', var_name='Logistic Regression Risk', 
                          #plot_title='Logistic Regression Class Breakdown', bins=100)


# In[100]:


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


# In[101]:


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


# In[102]:


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = '#ED0000FF', label = 'Model')#crimson颜色也好看
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = '#ED0000FF', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    #ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_ylim(-0.1,0.6)
    ax.set_xlabel(
        xlabel = 'Threshold Probability')#fontdict= {'family': 'Times New Roman', 'fontsize': 15}设置字体
    ax.set_ylabel(
        ylabel = 'Net Benefit')
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax


# In[103]:


from sklearn.metrics import confusion_matrix


# In[109]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 9, 8
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率


# In[110]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, pred_proba_lr[:, 1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("Logistic Regression")
plt.figure(figsize=(10, 10), dpi=600)
#plt.figure(dpi=300,figsize=(24,8))
plt.show()


# ROC曲线+95%CI Bootstrap方法

# In[104]:


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


# In[105]:


y = y_test
scores_lr = pred_proba_lr[:, 1]
statistics_lr = bootstrap_auc(y,scores_lr,[0,1])
print("均值:",np.mean(statistics_lr,axis=1))
print("最大值:",np.max(statistics_lr,axis=1))
print("最小值:",np.min(statistics_lr,axis=1))


# ROC曲线+95%CI 方法2

# In[113]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# # XGboot

# In[106]:


from xgboost.sklearn import XGBClassifier


# In[107]:


Xgbc_clf=XGBClassifier(random_state=42)  #Xgbc
plot_learning_curves(Xgbc_clf, data_train_X, y_train)


# max_depth = 5 ：这应该在3-10之间。我从5开始，但你也可以选择不同的数字。4-6可以是很好的起点。
# min_child_weight = 1 ：选择较小的值是因为它是高度不平衡的类问题，并且叶节点可以具有较小的大小组。
# gamma = 0.1 ：也可以选择较小的值，如0.1-0.2来启动。无论如何，这将在以后进行调整。
# subsample，colsample_bytree = 0.8：这是一个常用的使用起始值。典型值介于0.5-0.9之间。
# scale_pos_weight = 1：由于高级别的不平衡。
# colsample_bytree = 0.5,gamma=0.2

# In[108]:


param_distribs = {
     'n_estimators': stats.randint(low=20, high=120),      
    'max_depth': stats.randint(low=1, high=100),
    'min_child_weight': stats.randint(low=1, high=100)
    }
Xgbc_clf=XGBClassifier(random_state=42,learning_rate=0.125,use_label_encoder=False)
Xgbc_search = RandomizedSearchCV(Xgbc_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
Xgbc_gs=Xgbc_search.fit(data_train_X, y_train)


# In[109]:


print(Xgbc_gs.best_score_)


# In[110]:


print(Xgbc_gs.best_params_)


# In[111]:


cv_rlt = Xgbc_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[112]:


Xgbc_clf = Xgbc_search.best_estimator_
Xgbc_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/Xgbc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(Xgbc_clf, f)


# In[113]:


Xgbc_clf


# In[114]:


plot_learning_curves(Xgbc_clf, data_train_X, y_train)


# # XGBoost-ROC

# In[115]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/Xgbc_clf_final_round.pkl', 'rb') as f:
    Xgbc_clf = pickle.load(f)
Xgbc_clf.fit(data_train_X, y_train)


# In[116]:


accu_Xgbc = accuracy_score(y_test, Xgbc_clf.predict(data_test_X))
round(accu_Xgbc,3)


# In[117]:


pd.crosstab(y_test, Xgbc_clf.predict(data_test_X))


# In[118]:


pred_proba_Xgbc = Xgbc_clf.predict_proba(data_test_X)


# In[119]:


fpr, tpr, _ = roc_curve(y_test, pred_proba_Xgbc[:, 1])
auc_Xgbc = roc_auc_score(y_test, pred_proba_Xgbc[:, 1])
round(auc_Xgbc,3)


# In[120]:


Xgbc_score = Xgbc_clf.score(data_test_X, y_test)
Xgbc_accuracy_score=accuracy_score(y_test,Xgbc_clf.predict(data_test_X))
Xgbc_preci_score=precision_score(y_test,Xgbc_clf.predict(data_test_X))
Xgbc_recall_score=recall_score(y_test,Xgbc_clf.predict(data_test_X))
Xgbc_f1_score=f1_score(y_test,Xgbc_clf.predict(data_test_X))
Xgbc_auc=roc_auc_score(y_test,pred_proba_Xgbc[:, 1])
print('Xgbc_accuracy_score: %f,Xgbc_preci_score: %f,Xgbc_recall_score: %f,Xgbc_f1_score: %f,Xgbc_auc: %f'
      %(Xgbc_accuracy_score,Xgbc_preci_score,Xgbc_recall_score,Xgbc_f1_score,Xgbc_auc))


# In[121]:


brier_score_loss(y_test, pred_proba_Xgbc[:, 1])


# In[122]:


print('loss Xgbc:', log_loss(y_test, pred_proba_Xgbc[:, 1]))


# In[123]:


Xgbc_results = [Xgbc_accuracy_score,Xgbc_preci_score,Xgbc_recall_score,Xgbc_f1_score,Xgbc_auc,brier_score_loss(y_test, pred_proba_Xgbc[:, 1]),log_loss(y_test, pred_proba_Xgbc[:, 1])]
Xgbc_results = pd.DataFrame(Xgbc_results)
Xgbc_results.columns = ['eXGBM']
Xgbc_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
Xgbc_results


# In[124]:


Xgbc_results.to_excel('F:/materi/12-SXD/postoperative infection/Xgbc_results.xlsx')


# In[125]:


plot_roc_curve(fpr, tpr, round(auc_Xgbc,3), Xgbc_clf)


# In[126]:


X_test['lr_pred_proba'] = pred_proba_Xgbc[:, 1]


# In[127]:


xgbm_his = X_test.join(y_test)


# In[128]:


xgbm_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-Xgbc.csv'.format(len(xgbm_his)), index=False)


# In[129]:


y_Xgbc = y_test
scores_Xgbc = pred_proba_Xgbc[:, 1]
statistics_Xgbc = bootstrap_auc(y_Xgbc,scores_Xgbc,[0,1])
print("均值:",np.mean(statistics_Xgbc,axis=1))
print("最大值:",np.max(statistics_Xgbc,axis=1))
print("最小值:",np.min(statistics_Xgbc,axis=1))


# In[130]:


Xgbc_his = X_test.join(y_test)
Xgbc_his.shape


# In[147]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, pred_proba_Xgbc[:, 1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("eXGBoosting Machine (External)")
#plt.figure(figsize=(10, 10), dpi=600)
plt.figure(dpi=300,figsize=(24,8))
plt.show()


# In[131]:


import shap


# In[132]:


#conda install shap


# In[133]:


shap.initjs()


# In[134]:


model = Xgbc_clf.fit(data_train_X, y_train)


# In[139]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(data_train_X)###修改此处为不同数据集里面进行运算哦！test


# In[140]:


image = shap.plots.force(shap_values[1])


# In[141]:


image


# In[142]:


shap.plots.beeswarm(shap_values, max_display=10)###


# # DecisionTreeClassifier

# In[143]:


from sklearn.tree import DecisionTreeClassifier


# In[144]:


tr_clf=DecisionTreeClassifier(random_state=42)  # 决策树模型
plot_learning_curves(tr_clf, data_train_X, y_train)


# In[145]:


param_distribs = {
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=50),
        'min_samples_split': stats.randint(low=2, high=200), 
        'min_samples_leaf': stats.randint(low=2, high=200)
    }
dt_clf = DecisionTreeClassifier(random_state=42,criterion='gini', splitter='best')
rnd_search = RandomizedSearchCV(dt_clf, param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
gsdt=rnd_search.fit(data_train_X, y_train)


# In[146]:


print(gsdt.best_score_)


# In[147]:


print(gsdt.best_params_)


# In[148]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[149]:


dt_clf = rnd_search.best_estimator_
dt_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/dt_clf_final_round.pkl', 'wb') as f:
    pickle.dump(dt_clf, f)


# In[150]:


plot_learning_curves(dt_clf, data_train_X, y_train)


# # Decision Tree ROC计算

# In[151]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/dt_clf_final_round.pkl', 'rb') as f:
    dt_clf = pickle.load(f)
dt_clf.fit(data_train_X, y_train)


# In[152]:


accu_dt = accuracy_score(y_test, dt_clf.predict(data_test_X))
round(accu_dt,3)


# In[153]:


pd.crosstab(y_test, dt_clf.predict(data_test_X))


# In[154]:


pred_proba_dt = dt_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(y_test, pred_proba_dt[:, 1])
auc_dt = roc_auc_score(y_test, pred_proba_dt[:, 1])
round(auc_dt,3)


# In[155]:


dt_score = dt_clf.score(data_test_X, y_test)
dt_accuracy_score=accuracy_score(y_test,dt_clf.predict(data_test_X))
dt_preci_score=precision_score(y_test,dt_clf.predict(data_test_X))
dt_recall_score=recall_score(y_test,dt_clf.predict(data_test_X))
dt_f1_score=f1_score(y_test,dt_clf.predict(data_test_X))
dt_auc=roc_auc_score(y_test,pred_proba_dt[:, 1])
print('dt_accuracy_score: %f,dt_preci_score: %f,dt_recall_score: %f,dt_f1_score: %f,dt_auc: %f'
      %(dt_accuracy_score,dt_preci_score,dt_recall_score,dt_f1_score,dt_auc))


# In[156]:


brier_score_loss(y_test, pred_proba_dt[:, 1])


# In[157]:


print('loss dt:', log_loss(y_test, pred_proba_dt[:, 1]))


# In[158]:


dt_results = [dt_accuracy_score,dt_preci_score,dt_recall_score,dt_f1_score,dt_auc,brier_score_loss(y_test, pred_proba_dt[:, 1]),log_loss(y_test, pred_proba_dt[:, 1])]
dt_results = pd.DataFrame(dt_results)
dt_results.columns = ['DT']
dt_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
dt_results


# In[159]:


dt_results.to_excel('F:/materi/12-SXD/postoperative infection/dt_results.xlsx')


# In[160]:


plot_roc_curve(fpr, tpr, round(auc_dt,3), dt_clf)


# In[161]:


X_test['lr_pred_proba'] = pred_proba_dt[:, 1]


# In[162]:


dt_his = X_test.join(y_test)


# In[163]:


dt_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-decision tree.csv'.format(len(dt_his)), index=False)


# In[164]:


y_dt = y_test
scores_dt = pred_proba_dt[:, 1]
statistics_dt = bootstrap_auc(y_dt,scores_dt,[0,1])
print("均值:",np.mean(statistics_dt,axis=1))
print("最大值:",np.max(statistics_dt,axis=1))
print("最小值:",np.min(statistics_dt,axis=1))


# In[171]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, pred_proba_dt[:, 1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("Decision Tree")
#plt.figure(figsize=(10, 10), dpi=600)
plt.figure(dpi=300,figsize=(24,8))
plt.show()


# # KNN机器学习算法

# In[165]:


from sklearn.neighbors import KNeighborsClassifier


# In[166]:


knn_clf=KNeighborsClassifier()  # 决策树模型
plot_learning_curves(knn_clf, data_train_X, y_train)


# In[167]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    { 'weights':['uniform'],
     'n_neighbors':[i for i in range (1,11)]##网上多为设置为11
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range (1,11)],##网上多为设置为11
        'p':[i for i in range(1,6)]
    }
]

grid_search_knn = GridSearchCV(knn_clf, param_grid, cv=5, n_jobs=-1, verbose=10)

grid_search_knn.fit(data_train_X, y_train)


# In[168]:


grid_search_knn.best_score_


# In[169]:


grid_search_knn.best_params_


# In[170]:


grid_search_knn.best_estimator_


# In[171]:


knn_clf = grid_search_knn.best_estimator_
knn_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/knn_clf_final_round.pkl', 'wb') as f:
    pickle.dump(knn_clf, f)


# In[172]:


plot_learning_curves(knn_clf, data_train_X, y_train)


# # KNN-ROC

# In[173]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/knn_clf_final_round.pkl', 'rb') as f:
    knn_clf = pickle.load(f)
knn_clf.fit(data_train_X, y_train)


# In[174]:


knn_clf


# In[175]:


accu_knn = accuracy_score(y_test, knn_clf.predict(data_test_X))
round(accu_knn,3)


# In[176]:


pd.crosstab(y_test, knn_clf.predict(data_test_X))


# In[177]:


pred_proba_knn = knn_clf.predict_proba(data_test_X)
fpr, tpr, _ = roc_curve(y_test, pred_proba_knn[:, 1])
auc_knn = roc_auc_score(y_test, pred_proba_knn[:, 1])
round(auc_knn,3)


# In[178]:


knn_score = knn_clf.score(data_test_X, y_test)
knn_accuracy_score=accuracy_score(y_test,knn_clf.predict(data_test_X))
knn_preci_score=precision_score(y_test,knn_clf.predict(data_test_X))
knn_recall_score=recall_score(y_test,knn_clf.predict(data_test_X))
knn_f1_score=f1_score(y_test,knn_clf.predict(data_test_X))
knn_auc=roc_auc_score(y_test,pred_proba_knn[:, 1])
print('knn_accuracy_score: %f,knn_preci_score: %f,knn_recall_score: %f,knn_f1_score: %f,knn_auc: %f'
      %(knn_accuracy_score,knn_preci_score,knn_recall_score,knn_f1_score,knn_auc))


# In[179]:


brier_score_loss(y_test, pred_proba_knn[:, 1])


# In[180]:


print('loss knn:', log_loss(y_test, pred_proba_knn[:, 1]))


# In[185]:


knn_results = [knn_accuracy_score,knn_preci_score,knn_recall_score,knn_f1_score,knn_auc,brier_score_loss(y_test, pred_proba_knn[:, 1]),log_loss(y_test, pred_proba_knn[:, 1])]
knn_results = pd.DataFrame(knn_results)
knn_results.columns = ['knn']
knn_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
knn_results


# In[186]:


knn_results.to_excel('F:/materi/12-SXD/postoperative infection/knn_results.xlsx')


# In[187]:


plot_roc_curve(fpr, tpr, round(auc_knn,3), knn_clf)


# In[188]:


X_test['lr_pred_proba'] = pred_proba_knn[:, 1]

knn_his = X_test.join(y_test)


# In[189]:


knn_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-knn.csv'.format(len(knn_his)), index=False)


# In[190]:


y_knn = y_test
scores_knn = pred_proba_knn[:, 1]
statistics_knn = bootstrap_auc(y_knn,scores_knn,[0,1])
print("均值:",np.mean(statistics_knn,axis=1))
print("最大值:",np.max(statistics_knn,axis=1))
print("最小值:",np.min(statistics_knn,axis=1))


# In[192]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, pred_proba_knn[:, 1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("K-Nearest Neighbor")
#plt.figure(figsize=(10, 10), dpi=600)
plt.figure(dpi=300,figsize=(24,8))
#plt.ylim(0, 0.5)
plt.show()


# # Neural Network

# In[205]:


from sklearn.neural_network import MLPClassifier


# In[206]:


nn_clf = MLPClassifier(random_state=42)
#plot_learning_curves(nn_clf, data_train_X, y_train)


# In[298]:


from keras.wrappers.scikit_learn import KerasRegressor###目前的存在的问题tensorflow 2.0版本以及以下才有compat


# In[299]:


import tensorflow as tf2


# In[300]:


#conda install tensorflow==2.0.0


# In[301]:


#from tensorflow import compat


# In[302]:


#conda install tensorflow


# In[303]:


from keras.wrappers.scikit_learn import KerasClassifier


# In[212]:


import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
# 跟原来的在理论上没有区别


# In[213]:


#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()


# In[214]:


#conda install tensorflow ##加载工具包，最好使用conda pip install加载不上的，用这个可以加载上去。


# In[ ]:


def create_network(optimizer='rmsprop', neurons=16, learning_rate=0.001):
    
    # Start Artificial Neural Network
    network = Sequential()
    
    # Adding the input layer and the first hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the second hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the third hidden layer
    network.add(Dense(units = neurons, 
                  activation = tf.keras.layers.LeakyReLU(alpha=0.3)))

    # Adding the output layer
    network.add(Dense(units = 1))

    ###############################################
    # Add optimizer with learning rate
    if optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('optimizer {} unrecognized'.format(optimizer))
    ##############################################    

    # Compile NN
    network.compile(optimizer = opt, 
                loss = 'mean_squared_error', 
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    # Return compiled network
    return network

# Wrap Keras model so it can be used by scikit-learn
ann = KerasRegressor(build_fn=create_network, verbose=0)

# Create hyperparameter space
epoch_values = [10, 25, 50, 100, 150, 200]
batches = [10, 20, 30, 40, 50, 100, 1000]
optimizers = ['rmsprop', 'adam', 'SGD']
neuron_list = [16, 32, 64, 128, 256]
lr_values = [0.001, 0.01, 0.1, 0.2, 0.3]

# Create hyperparameter options
hyperparameters = dict(
    epochs=epoch_values, 
    batch_size=batches, 
    optimizer=optimizers, 
    neurons=neuron_list,
    learning_rate=lr_values)

# Create grid search
# cv=5 is the default 5-fold
grid = GridSearchCV(estimator=ann, cv=5, param_grid=hyperparameters)

# Fit grid search
grid_result = grid.fit(data_train_X, y_train)


# In[219]:


from keras.optimizers import SGD


# In[ ]:


# 构建模型的函数
def create_model(learn_rate=0.01, momentum=0):
    # 创建模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 为了复现，设置随机种子
seed = 7
np.random.seed(seed)

# 加载数据
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 切分数据为输入 X 和输出 Y
X = dataset[:,0:8]
Y = dataset[:,8]

# 创建模型，使用到了上一步找出的 epochs、batch size 最优参数
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=80, verbose=0)
# 定义网格搜索参数
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)

# 总结结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[207]:


nn_clf = MLPClassifier(random_state=42,activation='relu',alpha=0.0001,batch_size='auto',beta_1=0.9, beta_2=0.999, 
                       early_stopping=False,epsilon=1e-08,hidden_layer_sizes=(100),
                       learning_rate='constant', learning_rate_init=0.001,max_iter=200, momentum=0.9, 
                       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,shuffle=True,  
                       tol=0.0001, validation_fraction=0.1,verbose=False, warm_start=False)


# In[208]:


nn_clf.fit(data_train_X, y_train)


# In[209]:


#rf_clf = rnd_search.best_estimator_
#rf_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/nn_clf_final_round.pkl', 'wb') as f:
    pickle.dump(nn_clf, f)


# In[210]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/nn_clf_final_round.pkl', 'rb') as f:
    nn_clf = pickle.load(f)
nn_clf.fit(data_train_X, y_train)


# In[211]:


nn_clf_y_pre=nn_clf.predict(data_test_X)
nn_clf_y_proba=nn_clf.predict_proba(data_test_X)


# In[212]:


pd.crosstab(y_test, nn_clf.predict(data_test_X))


# In[213]:


#from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve


# In[214]:


nn_clf_accuracy_score=accuracy_score(y_test,nn_clf_y_pre)
nn_clf_preci_score=precision_score(y_test,nn_clf_y_pre)
nn_clf_recall_score=recall_score(y_test,nn_clf_y_pre)
nn_clf_f1_score=f1_score(y_test,nn_clf_y_pre)
nn_clf_auc=roc_auc_score(y_test,nn_clf_y_proba[:,1])
print('nn_clf_accuracy_score: %f,nn_clf_preci_score: %f,nn_clf_recall_score: %f,nn_clf_f1_score: %f,nn_clf_auc: %f'
      %(nn_clf_accuracy_score,nn_clf_preci_score,nn_clf_recall_score,nn_clf_f1_score,nn_clf_auc))


# In[215]:


brier_score_loss(y_test, nn_clf_y_proba[:,1])


# In[216]:


print('loss nn:', log_loss(y_test, nn_clf_y_proba[:,1]))


# In[217]:


nn_clf_results = [nn_clf_accuracy_score,nn_clf_preci_score,nn_clf_recall_score,nn_clf_f1_score,nn_clf_auc,brier_score_loss(y_test, nn_clf_y_proba[:, 1]),log_loss(y_test, nn_clf_y_proba[:, 1])]
nn_clf_results = pd.DataFrame(nn_clf_results)
nn_clf_results.columns = ['nn_clf']
nn_clf_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
nn_clf_results


# In[218]:


nn_clf_results.to_excel('F:/materi/12-SXD/postoperative infection/nn_results.xlsx')


# In[219]:


X_test['lr_pred_proba'] = nn_clf_y_proba[:,1]


# In[220]:


nn_his = X_test.join(y_test)


# In[221]:


nn_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-nn.csv'.format(len(nn_his)), index=False)


# In[222]:


fpr, tpr, _ = roc_curve(y_test, nn_clf_y_proba[:,1])
auc_nn = roc_auc_score(y_test, nn_clf_y_proba[:,1])


# In[223]:


round(auc_nn,3)


# In[224]:


plot_roc_curve(fpr, tpr, round(auc_nn,3), nn_clf)


# In[225]:


y_nn = y_test
scores_nn = nn_clf_y_proba[:,1]
statistics_nn = bootstrap_auc(y_nn,scores_nn,[0,1])
print("均值:",np.mean(statistics_nn,axis=1))
print("最大值:",np.max(statistics_nn,axis=1))
print("最小值:",np.min(statistics_nn,axis=1))


# In[234]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, nn_clf_y_proba[:,1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("Neural Network")
#plt.figure(figsize=(10, 10), dpi=600)
#plt.figure(dpi=300,figsize=(24,8))
plt.figure(dpi=300,figsize=(24,8))
plt.tight_layout()

plt.rcParams['savefig.dpi'] = 1200 #图片像素
plt.rcParams['figure.dpi'] = 1200 #分辨率
#plt.figure(figsize=(8,10))
plt.show()


# # Gradient boosting classifier---随机搜索模型调参RandomizedSearchCV
# Gradient boosting classifier is an ensemble tree-based model that reduces the bias of the predictors.

# In[226]:


plot_learning_curves(GradientBoostingClassifier(random_state=42), data_train_X, y_train)


# In[227]:


param_distribs = {
        'n_estimators': stats.randint(low=80, high=200),
         'max_features': ['auto', 'log2'],
        'max_depth': stats.randint(low=1, high=100),
        'min_samples_split': stats.randint(low=2, high=200), 
        'min_samples_leaf': stats.randint(low=2, high=200),
    }

rnd_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), 
                                param_distributions=param_distribs, return_train_score=True,
                                n_iter=100, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
# this will take a long time
gsgbm = rnd_search.fit(data_train_X, y_train)


# In[228]:


print(gsgbm.best_score_)


# In[229]:


print(gsgbm.best_params_)


# In[230]:


cv_rlt = rnd_search.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[231]:


gbm_clf = rnd_search.best_estimator_
gbm_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/gbm_clf_final_round.pkl', 'wb') as f:
    pickle.dump(gbm_clf, f)


# In[232]:


plot_learning_curves(gbm_clf, data_train_X, y_train)


# # Gradient boosting machine model-ROC

# In[233]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/gbm_clf_final_round.pkl', 'rb') as f:
    gbm_clf = pickle.load(f)
gbm_clf.fit(data_train_X, y_train)


# Accuracy scores

# In[234]:


accu_gbm = accuracy_score(y_test, gbm_clf.predict(data_test_X))


# In[235]:


round(accu_gbm,3)


# In[236]:


pd.crosstab(y_test, gbm_clf.predict(data_test_X))


# ROC and AUC

# In[ ]:





# In[289]:


pred_proba_gbm = gbm_clf.predict_proba(data_test_X)


# In[238]:


fpr, tpr, _ = roc_curve(y_test, pred_proba_gbm[:, 1])
auc_gbm = roc_auc_score(y_test, pred_proba_gbm[:, 1])


# In[239]:


round(auc_gbm,3)


# In[240]:


gbm_score = gbm_clf.score(data_test_X, y_test)
gbm_accuracy_score=accuracy_score(y_test,gbm_clf.predict(data_test_X))
gbm_preci_score=precision_score(y_test,gbm_clf.predict(data_test_X))
gbm_recall_score=recall_score(y_test,gbm_clf.predict(data_test_X))
gbm_f1_score=f1_score(y_test,gbm_clf.predict(data_test_X))
gbm_auc=roc_auc_score(y_test,pred_proba_gbm[:, 1])
print('gbm_accuracy_score: %f,gbm_preci_score: %f,gbm_recall_score: %f,gbm_f1_score: %f,gbm_auc: %f'
      %(gbm_accuracy_score,gbm_preci_score,gbm_recall_score,gbm_f1_score,gbm_auc))


# In[280]:


brier_score_loss(y_test, pred_proba_gbm[:, 1])


# In[281]:


print('loss gbm:', log_loss(y_test, pred_proba_gbm[:, 1]))


# In[287]:


gbm_results = [gbm_accuracy_score,gbm_preci_score,gbm_recall_score,gbm_f1_score,gbm_auc,brier_score_loss(y_test, pred_proba_gbm[:, 1]),log_loss(y_test, pred_proba_gbm[:, 1])]
gbm_results = pd.DataFrame(gbm_results)
gbm_results.columns = ['gbm']
gbm_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
gbm_results


# In[288]:


gbm_results.to_excel('F:/materi/12-SXD/postoperative infection/gbm_results.xlsx')


# In[290]:


plot_roc_curve(fpr, tpr, round(auc_gbm,3), gbm_clf)


# In[293]:


X_test['lr_pred_proba'] = pred_proba_gbm[:,1]
gbm_his = X_test.join(y_test)
gbm_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-gbm.csv'.format(len(nn_his)), index=False)


# In[292]:


y = y_test
scores_gbm = pred_proba_gbm[:, 1]
statistics_gbm = bootstrap_auc(y,scores_gbm,[0,1])
print("均值:",np.mean(statistics_gbm,axis=1))
print("最大值:",np.max(statistics_gbm,axis=1))
print("最小值:",np.min(statistics_gbm,axis=1))


# In[234]:


import shap


# In[235]:


shap.initjs()


# In[256]:


# train an model
model = gbm_clf.fit(data_train_X, y_train)


# In[257]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(data_test_X)###修改此处为不同数据集里面进行运算哦！！！

# visualize the first prediction's explanation
shap.waterfall_plot(explainer.base_values[0])
# In[238]:


shap.initjs()
# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0])


# In[245]:


shap.initjs()
# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[38])


# In[262]:


import os
plt.tight_layout()
#plt.savefig('images.png')
plt.savefig(os.path.join('E:/R code-LMX/Yimin-medical disputes/Models', 'GBM-julei.png'))
plt.rcParams['savefig.dpi'] = 1200 #图片像素
plt.rcParams['figure.dpi'] = 1200 #分辨率
plt.figure(figsize=(8,10))
plt.show()


# # Support vector machine classifier---网格搜索模型调参GridSearchCV
# Support vector machine classifier is a powerful classifier that works best on small to medium size complex data set. Our training set is medium size to SVMs.
# 
# plot the learning curve to find out where the default model is at

# Try Linear SVC fist

# In[241]:


plot_learning_curves(LinearSVC(loss='hinge', random_state=42), data_train_X, y_train)


# Try Polynomial kernel

# In[223]:


plot_learning_curves(SVC(kernel='poly', random_state=42), data_train_X, y_train)


# Try Gaussian RBF kernel

# In[224]:


plot_learning_curves(SVC(random_state=42), data_train_X, y_train)


# In[ ]:





# 第二调参

# In[242]:


hyperparameters = {
 "C": stats.uniform(0.001, 0.1),
 "gamma": stats.uniform(0, 0.5),
 'kernel': ('linear', 'rbf')
}
random = RandomizedSearchCV(estimator = SVC(probability=True), param_distributions = hyperparameters, n_iter = 100, 
                            cv = 5, return_train_score=True, random_state=42, n_jobs = -1)
gssvm = random.fit(data_train_X, y_train)


# In[243]:


print(gssvm.best_score_)


# In[244]:


print(gssvm.best_params_)


# In[245]:


cv_rlt = random.cv_results_
pd.DataFrame(sorted(list(zip(cv_rlt["mean_test_score"], cv_rlt['mean_train_score'], cv_rlt["params"])), 
                    key=lambda x: x[0], reverse=True), columns=['X-val scores', 'Train scores', 'Params'])


# In[246]:


svc_clf = random.best_estimator_
#svc_clf.fit(data_train_X, y_train)
with open('F:/materi/12-SXD/postoperative infection/svc_clf_final_round.pkl', 'wb') as f:
    pickle.dump(svc_clf, f)


# In[247]:


# best model is the default RBF kernal SVM
plot_learning_curves(svc_clf, data_train_X, y_train) 


# # SVM-ROC

# In[248]:


# Import model and retrain
with open('F:/materi/12-SXD/postoperative infection/svc_clf_final_round.pkl', 'rb') as f:
    svc_clf = pickle.load(f)
svc_clf.fit(data_train_X, y_train)


# In[249]:


accu_svc = accuracy_score(y_test, svc_clf.predict(data_test_X))


# In[250]:


round(accu_svc,3)


# In[251]:


pd.crosstab(y_test, svc_clf.predict(data_test_X))


# In[252]:


pred_proba_svc = svc_clf.predict_proba(data_test_X) 


# In[253]:


fpr, tpr, _ = roc_curve(y_test, pred_proba_svc[:, 1])
auc_svc = roc_auc_score(y_test, pred_proba_svc[:, 1])


# In[254]:


round(auc_svc,3)


# In[255]:


svc_score = svc_clf.score(data_test_X, y_test)
svc_accuracy_score=accuracy_score(y_test,svc_clf.predict(data_test_X))
svc_preci_score=precision_score(y_test,svc_clf.predict(data_test_X))
svc_recall_score=recall_score(y_test,svc_clf.predict(data_test_X))
svc_f1_score=f1_score(y_test,svc_clf.predict(data_test_X))
svc_auc=roc_auc_score(y_test,pred_proba_svc[:, 1])
print('svc_accuracy_score: %f,svc_preci_score: %f,svc_recall_score: %f,svc_f1_score: %f,svc_auc: %f'
      %(svc_accuracy_score,svc_preci_score,svc_recall_score,svc_f1_score,svc_auc))


# In[256]:


brier_score_loss(y_test, pred_proba_svc[:, 1])


# In[257]:


print('loss svc:', log_loss(y_test, pred_proba_svc[:, 1]))


# In[258]:


svc_results = [svc_accuracy_score,svc_preci_score,svc_recall_score,svc_f1_score,svc_auc,brier_score_loss(y_test, pred_proba_svc[:, 1]),log_loss(y_test, pred_proba_svc[:, 1])]
svc_results = pd.DataFrame(svc_results)
svc_results.columns = ['svc']
svc_results["Metrics"] = ["Accuracy", "Precise", "Recall", "F1 score", "AUC", "Brier score", "Log loss"]
svc_results


# In[259]:


svc_results.to_excel('F:/materi/12-SXD/postoperative infection/svc_results.xlsx')


# In[260]:


plot_roc_curve(fpr, tpr, round(auc_svc,3), svc_clf)


# In[261]:


X_test['lr_pred_proba'] = pred_proba_svc[:, 1]


# In[262]:


svm_his = X_test.join(y_test)


# In[263]:


svm_his.to_csv('F:/materi/12-SXD/postoperative infection/test_set_with_predictions-Support vector machine model.csv'.format(len(svm_his)), index=False)


# In[264]:


y = y_test
scores_svc = pred_proba_svc[:, 1]
statistics_svc = bootstrap_auc(y,scores_svc,[0,1])
print("均值:",np.mean(statistics_svc,axis=1))
print("最大值:",np.max(statistics_svc,axis=1))
print("最小值:",np.min(statistics_svc,axis=1))


# In[248]:


thresh_group = np.arange(0,1,0.01)
net_benefit_model = calculate_net_benefit_model(thresh_group, pred_proba_svc[:,1], y_test)
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
fig, ax = plt.subplots()
ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
# fig.savefig('fig1.png', dpi = 300)
plt.title("Support Vector Machine")
#plt.figure(figsize=(10, 10), dpi=600)
plt.figure(dpi=300,figsize=(24,8))
plt.show()


# In[300]:


# coding=utf-8
import matplotlib.pyplot as plt

game = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
              38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,
              76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])

plt.figure(figsize=(20, 10), dpi=600)##设置画布大小和像素参数。
#game = ['1', '2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
       # '3-G4', '3-G5', '总决赛-G1', '总决赛-G2', '总决赛-G3', '总决赛-G4', '总决赛-G5', '总决赛-G6']
#scores = statistics[1, :]
#plt.plot(game, scores)
plt.ylim(0.60, 1.00)

plt.plot(game, statistics_lr[0, :], c='#00468BFF', label="Logistic Regression (0.729 [0.654-0.803])",marker="s")
plt.plot(game, statistics_svc[0, :], c='#42B540FF', label="Support Vector Machine (0.930 [0.896-0.965])",marker="s")
plt.plot(game, statistics_nn[0, :], c='#FDAF91FF', label="Neural Network (0.944 [0.914-0.974])",marker="s")
plt.plot(game, statistics_gbm[0, :], c='#AD002AFF', label="Gradient Boosting Machine (0.986 [0.972-1.000])",marker="s")##ED0000FF
plt.plot(game, statistics_knn[0, :], c='#0099B4FF', label="K-Nearest Neighbor (0.962 [0.933-0.991])",marker="s")#linestyle='--',
plt.plot(game, statistics_dt[0, :], c='#925E9FFF', label="Decision Tree (0.871 [0.822-0.919])",marker="s")

#plt.plot(game, statistics_lightgbm[1, :], c='#925E9FFF', label="Light Gradient Boosting Machine (0.808, 95% CI: 0.778-0.831)")#linestyle='-.', 
#plt.plot(game, statistics_catgbm[1, :], c='#FFDC91FF', label="CatBoosting (0.785, 95% CI: 0.753-0.824)")
#plt.plot(game, statistics_ensemble[1, :], c='#1B1919FF', label="Ensemble Model (0.809, 95% CI: 0.772-0.839)")

plt.scatter(game, statistics_lr[0, :], c='#00468BFF')
plt.scatter(game, statistics_svc[0, :], c='#42B540FF')
plt.scatter(game, statistics_nn[0, :], c='#FDAF91FF')
plt.scatter(game, statistics_gbm[0, :], c='#AD002AFF')
plt.scatter(game, statistics_knn[0, :], c='#0099B4FF')
plt.scatter(game, statistics_dt[0, :], c='#925E9FFF')
#plt.scatter(game, statistics_lightgbm[1, :], c='#925E9FFF')#ED0000FF
#plt.scatter(game, statistics_catgbm[1, :], c='#FFDC91FF')
#plt.scatter(game, statistics_ensemble[1, :], c='#1B1919FF')


plt.legend(loc='best')
#plt.yticks(range(0, 50, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Number of bootstraps", fontdict={'size': 20})
plt.ylabel("AUC", fontdict={'size': 20})
plt.title("AUC (Bootstraps=100)", fontdict={'size': 20})

plt.legend(loc='lower right',fontsize=16)
#plt.xlabel('X-axis',fontproperties=font_set) #X轴标签
#plt.ylabel("Y-axis",fontproperties=font_set) #Y轴标签
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))#图注的位置设置：upper right，upper left，lower left，
#lower right，right，center left，center right，lower center，upper center，center

plt.yticks(size=15)#设置大小及加粗fontproperties='Times New Roman', ,weight='bold'
plt.xticks(size=15)

plt.show()

