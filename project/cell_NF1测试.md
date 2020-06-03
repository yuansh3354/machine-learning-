---------------------------------------

Title: "Machine Learning Project_001"

Date: "05/31/2020"

Reference : PMID: 29617658

Best Regards,

Yuan.SH

---------------------------------------

Please contact with me via follow way:

(a) E-mail :yuansh3354@163.com

(b) QQ :1044532817

(c) WeChat :Y1044532817

---



> 所有的代码在文献中都有提供，本次流程仅提供NF1基因突变分析
>
> 文献所使用的脚本为bash脚本
>
> 为了深刻理解和学习机器学习并向深度学习迈进
>
> 所以本文对该文献的基本脚本进行解析
>
> 仅供参考



### 数据



> 数据下载脚本在scripts/initialize



pancan_normalized_rnaseq.tsv：RNA_seq表达谱数据

mc3.v0.2.8.PUBLIC.maf : 针对来自33种癌症类型的超过10,000种癌症外显子组样本的变异识别项目

pancan_GISTIC_threshold.tsv：拷贝数变异信息

sampleset_freeze_version4_modify ：样本信息

seg_based_scores.tsv: 拷贝数变异负荷指数

### 代码

##### 1.加载模块

```python
### 加载模块
if True:
    from statsmodels.robust.scale import mad # 绝对中位差
    from sklearn.preprocessing import StandardScaler  # 数据的标准化处理
    from sklearn.pipeline import Pipeline  # 进行算法串联
    from sklearn.model_selection import GridSearchCV  # 网格搜索和交叉验证
    from sklearn.model_selection import train_test_split, cross_val_predict  # 数据分割和交叉验证
    from sklearn.linear_model import SGDClassifier  # 随机梯度下降分类器
    import seaborn as sns  # 数据可视化
    import matplotlib.pyplot as plt
    import csv
    import pandas as pd
    import warnings
    import sys
    import os
    wkd = '/Users/yuansh/Desktop/machine_learning/'
    os.chdir(wkd)
    # 绝对中位差: 一种评估数据离差指标,类似于标准差(标准差对异常点极其敏感)
    # 一般情况下,符合正态分布的数据使用标准差判断数据离差程度比较合理
    # 但是由于探索的是rnaseq表达谱(泊松分布),因此使用MAD来评估数据离散程度
		# 但是由于探索的是rnaseq表达谱(泊松分布),因此使用MAD来评估数据离散程度
		# 网格搜索是一种搜索最佳超参数的方法
		# 主要用于离散形变量的搜索
		# 对应的连续性超参数的搜索方法
		# 是 RandomizedSearchCV
```

##### 2.数据预处理

###### 拷贝数变异数据处理

1. 位置ID和染色体信息
2. 将患者ID整合成与表达谱一致的ID
3. 拷贝数变异信息包：拷贝数超缺失，拷贝数缺失，无变异，拷贝数增加，拷贝数超增加。仅保留拷贝数超缺失和拷贝数超增加样本为loss和gain

```python
if False:
    copy_input_file = os.path.join(wkd + 'data', 'raw', 'pancan_GISTIC_threshold.tsv')
    copy_loss_file = os.path.join(wkd + 'data', 'copy_number_loss_status.tsv.gz')
    copy_gain_file = os.path.join(wkd + 'data', 'copy_number_gain_status.tsv.gz')
    sample_freeze_file = os.path.join(wkd + 'data', 'sample_freeze.tsv')
    # Load data
    copy_thresh_df = pd.read_table(copy_input_file, index_col=0) # 拷贝数表达谱
    copy_thresh_df.drop(['Locus ID', 'Cytoband'], axis=1, inplace=True) # 删除ID和染色体位置信息
    copy_thresh_df.columns = copy_thresh_df.columns.str[0:15] # TCGA样本编号的15个字符具有唯一标识性
    sample_freeze_df = pd.read_table(sample_freeze_file) # 导入各个样本的信息
    # subset data 
    copy_thresh_df = copy_thresh_df.T # 行样本,列基因
    intersect = list(set(sorted(sample_freeze_df['SAMPLE_BARCODE'])).intersection(set(copy_thresh_df.index))) # 提取交集样本
    intersect = sorted(intersect)
    copy_thresh_df = copy_thresh_df.loc[intersect]
    copy_thresh_df = copy_thresh_df.fillna(0)
    copy_thresh_df = copy_thresh_df.astype(int)
    # 拷贝数表达谱中有5个值-2,-1,0,1,2
    # 只保留-2 (loss) 和 2 (gain)
    # loss
    copy_loss_df = copy_thresh_df.replace(to_replace=[1, 2, -1], value=0)
    copy_loss_df.replace(to_replace=-2, value=1, inplace=True)
    copy_loss_df.to_csv(copy_loss_file, sep='\t', compression='gzip')
    # gain
    copy_gain_df = copy_thresh_df.replace(to_replace=[-1, -2, 1], value=0)
    copy_gain_df.replace(to_replace=2, value=1, inplace=True)
    copy_gain_df.to_csv(copy_gain_file, sep='\t', compression='gzip')
```

###### 基因表达谱数据处理

1. 删除基因名中的奇奇怪怪的符号
2. 比对freeze样本
3. 删除重复基因

```python
### 基因表达谱处理
if False :
    rna_file = os.path.join(wkd + 'data', 'raw', 'pancan_normalized_rnaseq.tsv')
    mut_file = os.path.join(wkd + 'data', 'raw', 'mc3.v0.2.8.PUBLIC.maf.gz')
    sample_freeze_file = os.path.join(wkd + 'data', 'raw',
                                      'sampleset_freeze_version4_modify.csv')
    rna_out_file = os.path.join(wkd + 'data', 'pancan_rnaseq_freeze.tsv.gz')
    mut_out_file = os.path.join(wkd + 'data', 'pancan_mutation_freeze.tsv.gz')
    freeze_out_file = os.path.join(wkd + 'data', 'sample_freeze.tsv')
    burden_out_file = os.path.join(wkd + 'data', 'mutation_burden_freeze.tsv')
    # Load Datasets
    rnaseq_df = pd.read_table(rna_file, index_col=0)
    mutation_df = pd.read_table(mut_file)
    sample_freeze_df = pd.read_csv(sample_freeze_file)
    # Process RNAseq file 
    rnaseq_df.index = rnaseq_df.index.map(lambda x: x.split('|')[0])
    rnaseq_df.columns = rnaseq_df.columns.str.slice(start=0, stop=15)
    rnaseq_df = rnaseq_df.drop('?').fillna(0).sort_index(axis=1)
    # clean data
    indexs = rnaseq_df.index.drop_duplicates(keep = False) # 删除重复基因
    rnaseq_df = rnaseq_df.loc[indexs].T
    # extract sample 
    freeze_barcodes = set(sample_freeze_df.SAMPLE_BARCODE)
    freeze_barcodes = freeze_barcodes.intersection(set(rnaseq_df.index))
    # add annotion information
    mutation_df = mutation_df.assign(PATIENT_BARCODE=mutation_df
                                     .Tumor_Sample_Barcode
                                     .str.slice(start=0, stop=12))
    mutation_df = mutation_df.assign(SAMPLE_BARCODE=mutation_df
                                     .Tumor_Sample_Barcode
                                     .str.slice(start=0, stop=15))
    # 导出数据
    rnaseq_df = rnaseq_df.loc[freeze_barcodes, :]
    rnaseq_df = rnaseq_df[~rnaseq_df.index.duplicated()]
    rnaseq_df.to_csv(rna_out_file, sep='\t', compression='gzip')
```

###### 染色体突变信息处理

1. 提取满足条件的染色体突变类型
2. 生存mutation 矩阵
3. 计算各个样本的肿瘤突变负荷

```python
### 染色体突变信息处理
if False:
  # 只提取满足条件的突变类型
    mutations = {
        'Frame_Shift_Del',
        'Frame_Shift_Ins',
        'In_Frame_Del',
        'In_Frame_Ins',
        'Missense_Mutation',
        'Nonsense_Mutation',
        'Nonstop_Mutation',
        'RNA',
        'Splice_Site',
        'Translation_Start_Site',
    }
    
    # query 提取满足条件的染色体变异
    # query 使用描述语句代替代码
    # 没什么屁用,就是让代码看的简洁一点
    mut_pivot = (mutation_df.query("Variant_Classification in @mutations")
                            .groupby(['SAMPLE_BARCODE', 'Chromosome',
                                      'Hugo_Symbol'])
                            .apply(len).reset_index()
                            .rename(columns={0: 'mutation'}))
    # 转换突变(0-1)矩阵
    mut_pivot = (mut_pivot.pivot_table(index='SAMPLE_BARCODE',
                                       columns='Hugo_Symbol', values='mutation',
                                       fill_value=0)
                          .astype(bool).astype(int))
    # 删除没有突变信息的样本
    freeze_barcodes = freeze_barcodes.intersection(set(mut_pivot.index))
    mut_pivot = mut_pivot.loc[freeze_barcodes, :]
    mut_pivot = mut_pivot.astype(int)
    mut_pivot.to_csv(mut_out_file, sep='\t', compression='gzip')
    # 计算基因突变负荷
    burden_df = mutation_df[mutation_df['Variant_Classification'].isin(mutations)]
    burden_df = burden_df.groupby('SAMPLE_BARCODE').apply(len)
    burden_df = np.log10(burden_df)
    burden_df = burden_df.loc[freeze_barcodes]
    burden_df = burden_df.fillna(0)
    burden_df = pd.DataFrame(burden_df, columns=['log10_mut'])
    burden_df.to_csv(burden_out_file, sep='\t')
    # Write out finalized and subset sample freeze file
    sample_freeze_df = sample_freeze_df[sample_freeze_df.SAMPLE_BARCODE
                                                        .isin(freeze_barcodes)]
    sample_freeze_df.to_csv(freeze_out_file, sep='\t')


```

##### Step 1 定义函数

```python
### 定义函数
# 获取计算ROC
def get_threshold_metrics(y_true, y_pred, drop_intermediate=False,
                          disease='all'):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments:
    y_true - an array of gold standard mutation status
    y_pred - an array of predicted mutation status
    disease - a string that includes the corresponding TCGA study acronym

    Output:
    dict of AUROC, AUPR, pandas dataframes of ROC and PR data, and cancer-type
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    roc_columns = ['fpr', 'tpr', 'threshold']
    pr_columns = ['precision', 'recall', 'threshold']

    if drop_intermediate:
        # zip 函数类似于R语言的rbind
        # 用法还是有点差异的
        # 在python3.x版本中,zip类得用list 或者dict 调用
        roc_items = zip(roc_columns,
                        roc_curve(y_true, y_pred, drop_intermediate=False))
        # 这里注意一下,roc_curve 返回的是tuple类型的
        # tumple 类 和list 类 进行zip时会返回dict类
    else:
        roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
        
    roc_df = pd.DataFrame.from_dict(dict(roc_items))
    prec, rec, thresh = precision_recall_curve(y_true, y_pred) #精确率召回率曲线
    # prec 精确率
    # rec  召回率
    # thresh 阈值
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    # 这里就有点没搞明白,他为什么要多此一举的多做一步这个
    # pr_df = pd.DataFrame.from_records([prec, rec, thresh]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average='weighted') # 计算auc和roc 得分
    aupr = average_precision_score(y_true, y_pred, average='weighted')

    return {'auroc': auroc, 'aupr': aupr, 'roc_df': roc_df,
            'pr_df': pr_df, 'disease': disease}

# 整合拷贝数变异信息
def integrate_copy_number(y, cancer_genes_df, genes, loss_df, gain_df,
                          include_mutation=True):
    """
    Function to integrate copy number data to define gene activation or gene
    inactivation events. Copy number loss results in gene inactivation events
    and is important for tumor suppressor genes while copy number gain results
    in gene activation events and is important for oncogenes.

    Arguments:
    y - pandas dataframe samples by genes where a 1 indicates event
    cancer_genes_df - a dataframe listing bona fide cancer genes as defined by
                      the 20/20 rule in Vogelstein et al. 2013
    genes - the input list of genes to build the classifier for
    loss_df - a sample by gene dataframe listing copy number loss events
    gain_df - a sample by gene dataframe listing copy number gain events
    include_mutation - boolean to decide to include mutation status
    """

    # Find if the input genes are in this master list
    # 提取基因信息
    genes_sub = cancer_genes_df[cancer_genes_df['Gene Symbol'].isin(genes)]

    # Add status to the Y matrix depending on if the gene is a tumor suppressor
    # or an oncogene. An oncogene can be activated with copy number gains, but
    # a tumor suppressor is inactivated with copy number loss
    # 判断基因属于抑癌基因还是原癌基因
    tumor_suppressor = genes_sub[genes_sub['Classification*'] == 'TSG']
    oncogene = genes_sub[genes_sub['Classification*'] == 'Oncogene']

    copy_loss_sub = loss_df[tumor_suppressor['Gene Symbol']]
    copy_gain_sub = gain_df[oncogene['Gene Symbol']]

    # Append to column names for visualization
    # 将基因突变信息和表达谱结合
    copy_loss_sub.columns = [col + '_loss' for col in copy_loss_sub.columns]
    copy_gain_sub.columns = [col + '_gain' for col in copy_gain_sub.columns]

    # Add columns to y matrix
    y = y.join(copy_loss_sub)
    y = y.join(copy_gain_sub)

    # Fill missing data with zero (measured mutation but not copy number)
    y = y.fillna(0)
    y = y.astype(int)
    
    # 将基因表达剔除
    if not include_mutation:
        y = y.drop(genes, axis=1)
    return y

# 扰动基因
def shuffle_columns(gene):
    """
    To be used in an `apply` pandas func to shuffle columns around a datafame
    Import only
    """
    import numpy as np
    return np.random.permutation(gene.tolist())


```

##### Step 2 bash脚本默认参数设置

> 这里要注意一下，把bash默认参数转移到python脚本内的时候
>
> 里面的action = 'stat_true/false' 设置默认的时候记得要取反
>
> 这样调用的时候在取反就行

```python
### 设置默认参数
if True:
    genes = 'NF1'  # 目标基因
    diseases = 'Auto'# 探索的肿瘤类型
    folds = 5 # 交叉验证
    drop = False # 是否从rna谱中移除目标基因
    drop_rasopathy = False # 移除ras 通路相关基因, 这里就是应用到了先验知识
    copy_number = False # 是否导入拷贝数变异
    filter_count = 15 #
    filter_prop = 0.05
    num_features_kept = 8000
    alphas = '0.1,0.13,0.15,0.18,0.2,0.25,0.3'# 列表推导式 [fun for val in collection if condition]
    l1_ratios = '0.15,0.155,0.16,0.2,0.25,0.3,0.4'# 正则化筛选
    alt_genes = 'None'
    alt_filter_count = 15
    alt_filter_prop = 0.05
    alt_diseases = 'Auto'
    alt_folder = 'Auto'
    remove_hyper = False
    keep_inter = False
    x_matrix ='raw'
    shuffled = False
    shuffled_before_training = False
    no_mutation = True
    drop_expression = False
    drop_covariates = False

```

##### Step 3 导入数据

```python
### 导入数据
if True:
    # 路径
    expr_file = os.path.join( wkd + 'data', 'pancan_rnaseq_freeze.tsv.gz')
    mut_file = os.path.join(wkd + 'data', 'pancan_mutation_freeze.tsv.gz')
    mut_burden_file = os.path.join(wkd + 'data', 'mutation_burden_freeze.tsv')
    sample_freeze_file = os.path.join(wkd + 'data', 'sample_freeze.tsv')
    copy_loss_file = os.path.join(wkd + 'data', 'copy_number_loss_status.tsv.gz')
    copy_gain_file = os.path.join(wkd + 'data', 'copy_number_gain_status.tsv.gz')
    vogel_file = os.path.join(wkd + 'data', 'vogelstein_cancergenes.tsv')
    # 导入
    sample_freeze = pd.read_table(sample_freeze_file, index_col=0) # 样本肿瘤类型
    mut_burden = pd.read_table(mut_burden_file) # 突变负荷
    rnaseq_full_df = pd.read_table(expr_file, index_col=0, compression='gzip') # rna表达谱
    mutation_df = pd.read_table(mut_file, index_col=0, compression='gzip') 
    copy_loss_df = pd.read_table(copy_loss_file, index_col=0)
    copy_gain_df = pd.read_table(copy_gain_file, index_col=0)
    cancer_genes = pd.read_table(vogel_file)
```

##### Step 4 分类器指定参数

> 这里是根据bash脚本

```python
### 分类器指定参数
# 测试NF1
if True:
    genes = 'NF1' 
    genes = genes.split(',')
    drop = bool(~drop)
    copy_number = bool(~copy_number)
    diseases = 'BLCA,COAD,GBM,LGG,LUAD,LUSC,OV,PCPG,SARC,SKCM,STAD,UCEC'
    diseases = diseases.split(',')  
    alphas = '0.1,0.13,0.15,0.18,0.2,0.25,0.3'
    alphas = [float(x) for x in alphas.split(',')] 
    l1_ratios = '0.15,0.155,0.16,0.2,0.25,0.3,0.4'
    l1_ratios = [float(x) for x in l1_ratios.split(',')]  
    remove_hyper = bool(~remove_hyper)
    alt_folder = wkd + 'classifiers/NF1' 
    keep_inter = bool(~keep_inter)
    shuffled = bool(~shuffled)
```

##### Step 5 分类器

```python
### 获取基因突变信息
common_genes = set(mutation_df.columns).intersection(genes)
common_genes = list(common_genes.intersection(rnaseq_full_df.columns))
y = mutation_df[common_genes] # 提取目标基因突变谱
"""
y
Out[51]: 
                 NF1
SAMPLE_BARCODE      
TCGA-IB-7885-01    0
TCGA-D1-A3DH-01    0
TCGA-06-0152-02    0
TCGA-CC-A7IF-01    0
TCGA-P3-A6T2-01    0
             ...
TCGA-A2-A3XU-01    0
TCGA-ET-A3DP-01    0
TCGA-AO-A126-01    0
TCGA-CJ-4912-01    0
TCGA-E8-A44M-01    0
"""



#### 将目标基因从表达谱中移除
if drop:
    rnaseq_full_df.drop(common_genes, axis=1, inplace=True)
"""
rnaseq_full_df
Out[53]: 
                         A1BG       A1CF  ...       ZZEF1        ZZZ3
TCGA-IB-7885-01     46.518400     8.3914  ...  1049.26000  569.606000
TCGA-D1-A3DH-01    210.621000     0.0000  ...  1638.53000  766.492000
TCGA-06-0152-02     30.753300     0.0000  ...  1448.17000  629.333000
TCGA-CC-A7IF-01  20471.800000  1514.6700  ...   402.73700  479.169000
TCGA-P3-A6T2-01     17.219100     0.0000  ...  1001.29000  859.234000
                      ...        ...  ...         ...         ...
TCGA-ET-A3DP-01    160.362000     0.0000  ...  1844.59000  731.126000
TCGA-AB-2986-03    781.630556     0.0000  ...   970.72739  793.917523
TCGA-AO-A126-01    599.949000     0.0000  ...   600.35500  955.595000
TCGA-CJ-4912-01    285.939000     1.1760  ...  2112.01000  599.735000
TCGA-E8-A44M-01    168.887000     0.0000  ...  1483.46000  804.263000

[9122 rows x 20499 columns]
"""



#### 将RAS通路相关的基因移除
if drop_rasopathy:
    rasopathy_genes = set(['BRAF', 'CBL', 'HRAS', 'KRAS', 'MAP2K1', 'MAP2K2',
                           'NF1', 'NRAS', 'PTPN11', 'RAF1', 'SHOC2', 'SOS1',
                           'SPRED1', 'RIT1'])
    rasopathy_drop = list(rasopathy_genes.intersection(rnaseq_full_df.columns))
    rnaseq_full_df.drop(rasopathy_drop, axis=1, inplace=True)
    
    
    
#### 拷贝数变异信息
if copy_number:
    y = integrate_copy_number(y=y, cancer_genes_df=cancer_genes,
                              genes=common_genes, loss_df=copy_loss_df,
                              gain_df=copy_gain_df,
                              include_mutation=no_mutation)
"""
    y
Out[162]: 
                 NF1  NF1_loss
SAMPLE_BARCODE                
TCGA-IB-7885-01    0         0
TCGA-D1-A3DH-01    0         0
TCGA-06-0152-02    0         0
TCGA-CC-A7IF-01    0         0
TCGA-P3-A6T2-01    0         0
             ...       ...
TCGA-A2-A3XU-01    0         0
TCGA-ET-A3DP-01    0         0
TCGA-AO-A126-01    0         0
TCGA-CJ-4912-01    0         0
TCGA-E8-A44M-01    0         0

[9062 rows x 2 columns]
"""



#### 样本信息合并
y = y.assign(total_status=y.max(axis=1))
y = y.reset_index().merge(sample_freeze,
                          how='left').set_index('SAMPLE_BARCODE')
"""
y
['NF1', 'NF1_loss', 'total_status', 'PATIENT_BARCODE', 'DISEASE','SUBTYPE']
其中
NF1 : 基因突变状态
NF1_loss : 拷贝数变异
total_status : 整合基因突变或者缺失/拷贝 # 这一列主要是用来定义患者的label
"""


#### 根据不同种类疾病进行信息整理
count_df = y.groupby('DISEASE').sum() # 计算不同疾病基因突变总数
prop_df = count_df.divide(y['DISEASE'].value_counts(sort=False).sort_index(),
                          axis=0) # 计算频率

count_table = count_df.merge(prop_df, left_index=True, right_index=True,
                             suffixes=('_count', '_proportion'))
count_table.to_csv("count_table_file.csv")



#### 提取数量信息和比例信息
mut_count = count_df['total_status']
prop = prop_df['total_status']



#### 过滤疾病
if diseases[0] == 'Auto':
    filter_disease = (mut_count > filter_count) & (prop > filter_prop)
    diseases = filter_disease.index[filter_disease].tolist()
    
    
    
#### 提取表达谱
y_df = y[y.DISEASE.isin(diseases)].total_status # 基因突变信息
common_samples = list(set(y_df.index) & set(rnaseq_full_df.index))
y_df = y_df.loc[common_samples]
rnaseq_df = rnaseq_full_df.loc[y_df.index, :]



#### 过滤超高肿瘤突变负担数据
if remove_hyper:
    burden_filter = mut_burden['log10_mut'] < 5 * mut_burden['log10_mut'].std()
    mut_burden = mut_burden[burden_filter]
   
    
    
#### 构建肿瘤突变负担,基因突变表
y_matrix = mut_burden.merge(pd.DataFrame(y_df), right_index=True,
                            left_on='SAMPLE_BARCODE')\
    .set_index('SAMPLE_BARCODE')



#### 构建哑变量矩阵
y_sub = y.loc[y_matrix.index]['DISEASE']
covar_dummy = pd.get_dummies(sample_freeze['DISEASE']).astype(int) # 生成稀疏矩阵
covar_dummy.index = sample_freeze['SAMPLE_BARCODE']
covar = covar_dummy.merge(y_matrix, right_index=True, left_index=True)
covar = covar.drop('total_status', axis=1)

#### 构建信息矩阵
y_df = y_df.loc[y_sub.index] # 基因突变矩阵
strat = y_sub.str.cat(y_df.astype(str))  # 疾病类型
x_df = rnaseq_df.loc[y_df.index, :] # 基因表达矩阵
"""
y_df
Out[188]: 
SAMPLE_BARCODE
TCGA-D1-A3DH-01    0
TCGA-06-0152-02    0
TCGA-HU-A4HD-01    0
TCGA-CM-4751-01    0
TCGA-34-5232-01    0
                  ..
TCGA-HU-8608-01    0
TCGA-MP-A4T6-01    0
TCGA-CS-6666-01    0
TCGA-EB-A5SH-06    0
TCGA-55-6979-01    0
Name: total_status, Length: 4101, dtype: int64

x_df
Out[190]: 
                       A1BG        A1CF  ...        ZZEF1         ZZZ3
SAMPLE_BARCODE                           ...                          
TCGA-D1-A3DH-01  210.621000    0.000000  ...  1638.530000   766.492000
TCGA-06-0152-02   30.753300    0.000000  ...  1448.170000   629.333000
TCGA-HU-A4HD-01   39.561156  229.015595  ...  1529.500692   671.344706
TCGA-CM-4751-01   28.793400  185.558000  ...  1336.380000   494.515000
TCGA-34-5232-01   50.556600    0.000000  ...   958.373000   888.427000
                    ...         ...  ...          ...          ...
TCGA-HU-8608-01   24.873672    1.574215  ...  2472.865696  1060.150193
TCGA-MP-A4T6-01  371.111000    0.000000  ...  1018.120000   292.406000
TCGA-CS-6666-01   48.425200    0.000000  ...  1748.730000   983.145000
TCGA-EB-A5SH-06  776.749000    0.000000  ...  2351.910000   195.562000
TCGA-55-6979-01   73.161100    0.000000  ...   956.971000   596.169000

strat
Out[189]: 
SAMPLE_BARCODE
TCGA-D1-A3DH-01    UCEC0
TCGA-06-0152-02     GBM0
TCGA-HU-A4HD-01    STAD0
TCGA-CM-4751-01    COAD0
TCGA-34-5232-01    LUSC0
 
TCGA-HU-8608-01    STAD0
TCGA-MP-A4T6-01    LUAD0
TCGA-CS-6666-01     LGG0
TCGA-EB-A5SH-06    SKCM0
TCGA-55-6979-01    LUAD0
Name: DISEASE, Length: 4101, dtype: object
"""



#### 提取候选基因
if x_matrix == 'raw':
    med_dev = pd.DataFrame(mad(x_df), index=x_df.columns) # 计算各个基因的绝对中位差
    # 提取绝对中位差最大的8000个基因,这里的话绝对中位差越大,数据间分离的越散,原则上差异越大
    mad_genes = med_dev.sort_values(by=0, ascending=False)\
                       .iloc[0:num_features_kept].index.tolist()
    x_df = x_df.loc[:, mad_genes]

#### 数据标准化
fitted_scaler = StandardScaler().fit(x_df) # 减均值除方差
x_df_update = pd.DataFrame(fitted_scaler.transform(x_df),
                           columns=x_df.columns)
x_df_update.index = x_df.index
# 将表达谱和哑变量合并
x_df = x_df_update.merge(covar, left_index=True, right_index=True)

#### 移除干扰基因
if drop_expression:
    x_df = x_df.iloc[:, num_features_kept:]
elif drop_covariates:
    x_df = x_df.iloc[:, 0:num_features_kept]
    
    
    
#### 数据扰动
if shuffled_before_training:
    # Shuffle genes
    x_train_genes = x_df.iloc[:, range(num_features_kept)]
    rnaseq_shuffled_df = x_train_genes.apply(shuffle_columns, axis=1,
                                             result_type='broadcast')

    x_train_cov = x_df.iloc[:, num_features_kept:]
    x_df = pd.concat([rnaseq_shuffled_df, x_train_cov], axis=1)

#### 根据肿瘤类型拆分样本
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=strat)
'''
x_df
Out[207]: 
                     ACTB     GAPDH    COL1A1  ...  UCS  UVM  log10_mut
SAMPLE_BARCODE                                 ...                     
TCGA-D1-A3DH-01 -0.408990  0.859015 -0.239816  ...    0    0         91
TCGA-06-0152-02  0.307594 -0.247356 -0.298242  ...    0    0         64
TCGA-HU-A4HD-01 -0.384406 -0.404397 -0.150909  ...    0    0        161
TCGA-CM-4751-01  1.606875 -0.211687 -0.118594  ...    0    0         89
TCGA-34-5232-01  0.013187  0.448416 -0.262298  ...    0    0        165
                  ...       ...       ...  ...  ...  ...        ...
TCGA-HU-8608-01 -0.377149 -0.572526 -0.150599  ...    0    0        165
TCGA-MP-A4T6-01 -0.355383 -0.870895 -0.308946  ...    0    0         58
TCGA-CS-6666-01 -0.330968 -0.575743 -0.339679  ...    0    0         43
TCGA-EB-A5SH-06  0.558247 -0.205666 -0.293190  ...    0    0         89
TCGA-55-6979-01  1.148622 -0.806370  0.022443  ...    0    0        531

[4101 rows x 8034 columns]

y_df
Out[208]: 
SAMPLE_BARCODE
TCGA-D1-A3DH-01    0
TCGA-06-0152-02    0
TCGA-HU-A4HD-01    0
TCGA-CM-4751-01    0
TCGA-34-5232-01    0
                  ..
TCGA-HU-8608-01    0
TCGA-MP-A4T6-01    0
TCGA-CS-6666-01    0
TCGA-EB-A5SH-06    0
TCGA-55-6979-01    0
Name: total_status, Length: 4101, dtype: int64
'''

clf_parameters = {'classify__loss': ['log'],
                  'classify__penalty': ['elasticnet'],
                  'classify__alpha': alphas, 'classify__l1_ratio': l1_ratios}

# pipeline 以元组的形势进行存储
# steps [(name1,fuction1),(name2,function2)]
# 随机梯度回归
# SGDClassifier
# class_weight 自动平衡权重
# max_iter 每次训练损失函数使用的样本数
estimator = Pipeline(steps=[('classify', SGDClassifier(random_state=0,
                                                       class_weight='balanced',
                                                       loss='log',
                                                       max_iter=5,
                                                       tol=None))])
# 网格搜索
cv_pipeline = GridSearchCV(estimator=estimator, param_grid=clf_parameters,
                           n_jobs=-1, cv=folds, scoring='roc_auc',
                           return_train_score=True)
cv_pipeline.fit(X=x_train, y=y_train)
# concat 类似于R语言的cbind
cv_results = pd.concat([pd.DataFrame(cv_pipeline.cv_results_)
                          .drop('params', axis=1),
                        pd.DataFrame.from_records(cv_pipeline
                                                  .cv_results_['params'])],
                       axis=1)
#### 获取评分表
cv_score_mat = pd.pivot_table(cv_results, values='mean_test_score',
                              index='classify__l1_ratio',
                              columns='classify__alpha')
# 可视化
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization strength multiplier (alpha)')
ax.set_ylabel('Elastic net mixing parameter (l1_ratio)')
plt.tight_layout()
plt.savefig("cv_heatmap_file.pdf", dpi=600, bbox_inches='tight')
plt.close()

#### 预测及评估性能
y_predict_train = cv_pipeline.decision_function(x_train)
y_predict_test = cv_pipeline.decision_function(x_test)
metrics_train = get_threshold_metrics(y_train, y_predict_train,
                                      drop_intermediate=keep_inter)
metrics_test = get_threshold_metrics(y_test, y_predict_test,
                                     drop_intermediate=keep_inter)
#### 交叉验证
y_cv = cross_val_predict(cv_pipeline.best_estimator_, X=x_train, y=y_train,
                         cv=folds, method='decision_function')
metrics_cv = get_threshold_metrics(y_train, y_cv,
                                   drop_intermediate=keep_inter)

#### 扰动
if shuffled:
    # Shuffle genes
    x_train_genes = x_train.iloc[:, range(num_features_kept)]
    rnaseq_shuffled_df = x_train_genes.apply(shuffle_columns, axis=1,
                                             result_type='broadcast')

    x_train_cov = x_train.iloc[:, num_features_kept:]
    rnaseq_shuffled_df = pd.concat([rnaseq_shuffled_df, x_train_cov], axis=1)

    y_predict_shuffled = cv_pipeline.decision_function(rnaseq_shuffled_df)
    metrics_shuffled = get_threshold_metrics(y_train, y_predict_shuffled,
                                             drop_intermediate=keep_inter)

# 保存ROC结果
if keep_inter:
    train_roc = metrics_train['roc_df']
    train_roc = train_roc.assign(train_type='train')
    test_roc = metrics_test['roc_df']
    test_roc = test_roc.assign(train_type='test')
    cv_roc = metrics_cv['roc_df']
    cv_roc = cv_roc.assign(train_type='cv')
    full_roc_df = pd.concat([train_roc, test_roc, cv_roc])
    if shuffled:
        shuffled_roc = metrics_shuffled['roc_df']
        shuffled_roc = shuffled_roc.assign(train_type='shuffled')
        full_roc_df = pd.concat([full_roc_df, shuffled_roc])
    full_roc_df = full_roc_df.assign(disease='PanCan')
    
sns.set_style("whitegrid")
plt.figure(figsize=(3, 3))
total_auroc = {}
colors = ['blue', 'green', 'orange', 'grey']
idx = 0

metrics_list = [('Training', metrics_train), ('Testing', metrics_test),
                ('CV', metrics_cv)]
if shuffled:
    metrics_list += [('Random', metrics_shuffled)]



### 可视化
for label, metrics in metrics_list:

    roc_df = metrics['roc_df']
    plt.plot(roc_df.fpr, roc_df.tpr,
             label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']),
             linewidth=1, c=colors[idx])
    total_auroc[label] = metrics['auroc']
    idx += 1

plt.axis('equal')
plt.plot([0, 1], [0, 1], color='navy', linewidth=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
plt.title('')
plt.tick_params(labelsize=8)
lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=7.5)

plt.savefig('full_roc_file.pdf', dpi=600, bbox_extra_artists=(lgd,),
            bbox_inches='tight')
plt.close()

sns.set_style("whitegrid")
plt.figure(figsize=(3, 3))
total_auroc = {}
colors = ['blue', 'green', 'orange', 'grey']
idx = 0

metrics_list = [('Training', metrics_train), ('Testing', metrics_test),
                ('CV', metrics_cv)]
if shuffled:
    metrics_list += [('Random', metrics_shuffled)]

for label, metrics in metrics_list:

    roc_df = metrics['roc_df']
    plt.plot(roc_df.fpr, roc_df.tpr,
             label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']),
             linewidth=1, c=colors[idx])
    total_auroc[label] = metrics['auroc']
    idx += 1

plt.axis('equal')
plt.plot([0, 1], [0, 1], color='navy', linewidth=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
plt.title('')
plt.tick_params(labelsize=8)
lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=7.5)

plt.savefig('full_roc_file.pdf', dpi=600, bbox_extra_artists=(lgd,),
            bbox_inches='tight')
plt.close()

sns.set_style("whitegrid")
plt.figure(figsize=(3, 3))
total_aupr = {}
colors = ['blue', 'green', 'orange', 'grey']
idx = 0

metrics_list = [('Training', metrics_train), ('Testing', metrics_test),
                ('CV', metrics_cv)]
if shuffled:
    metrics_list += [('Random', metrics_shuffled)]

for label, metrics in metrics_list:
    pr_df = metrics['pr_df']
    plt.plot(pr_df.recall, pr_df.precision,
             label='{} (AUPR = {:.1%})'.format(label, metrics['aupr']),
             linewidth=1, c=colors[idx])
    total_aupr[label] = metrics['aupr']
    idx += 1

plt.axis('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=8)
plt.ylabel('Precision', fontsize=8)
plt.title('')
plt.tick_params(labelsize=8)
lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=7.5)

plt.savefig('full_pr_file.pdf', dpi=600, bbox_extra_artists=(lgd,),
            bbox_inches='tight')
plt.close()

disease_metrics = {}
# 根据不同的疾病类型,比较分类器
for disease in diseases:
    #提取疾病患者
    sample_sub = y_sub[y_sub == disease].index

    # Get true and predicted training labels
    y_disease_train = y_train[y_train.index.isin(sample_sub)]
    if y_disease_train.sum() < 1:
        continue
    y_disease_predict_train = y_predict_train[y_train.index.isin(sample_sub)]

    # Get true and predicted testing labels
    y_disease_test = y_test[y_test.index.isin(sample_sub)]
    if y_disease_test.sum() < 1:
        continue
    y_disease_predict_test = y_predict_test[y_test.index.isin(sample_sub)]

    # Get predicted labels for samples when they were in cross validation set
    # The true labels are y_pred_train
    y_disease_predict_cv = y_cv[y_train.index.isin(sample_sub)]

    # Get classifier performance metrics for three scenarios for each disease
    met_train_dis = get_threshold_metrics(y_disease_train,
                                          y_disease_predict_train,
                                          disease=disease,
                                          drop_intermediate=keep_inter)
    met_test_dis = get_threshold_metrics(y_disease_test,
                                         y_disease_predict_test,
                                         disease=disease,
                                         drop_intermediate=keep_inter)
    met_cv_dis = get_threshold_metrics(y_disease_train,
                                       y_disease_predict_cv,
                                       disease=disease,
                                       drop_intermediate=keep_inter)

    # Get predictions and metrics with shuffled gene expression
    if shuffled:
        y_dis_predict_shuf = y_predict_shuffled[y_train.index.isin(sample_sub)]
        met_shuff_dis = get_threshold_metrics(y_disease_train,
                                              y_dis_predict_shuf,
                                              disease=disease,
                                              drop_intermediate=keep_inter)

    if keep_inter:
        train_roc = met_train_dis['roc_df']
        train_roc = train_roc.assign(train_type='train')
        test_roc = met_test_dis['roc_df']
        test_roc = test_roc.assign(train_type='test')
        cv_roc = met_cv_dis['roc_df']
        cv_roc = cv_roc.assign(train_type='cv')
        full_dis_roc_df = train_roc.append(test_roc).append(cv_roc)

        if shuffled:
            shuffled_roc = met_shuff_dis['roc_df']
            shuffled_roc = shuffled_roc.assign(train_type='shuffled')
            full_dis_roc_df = full_dis_roc_df.append(shuffled_roc)

        full_dis_roc_df = full_dis_roc_df.assign(disease=disease)
        full_roc_df = full_roc_df.append(full_dis_roc_df)

    # Store results in disease indexed dictionary
    disease_metrics[disease] = [met_train_dis, met_test_dis, met_cv_dis]

    if shuffled:
        disease_metrics[disease] += [met_shuff_dis]

disease_auroc = {}
disease_aupr = {}
for disease, metrics_val in disease_metrics.items():

    labels = ['Training', 'Testing', 'CV', 'Random']
    met_list = []
    idx = 0
    for met in metrics_val:
        lab = labels[idx]
        met_list.append((lab, met))
        idx += 1

    disease_pr_sub_file = '{}_pred_{}.pdf'.format("disease_pr_file.pdf", disease)
    disease_roc_sub_file = '{}_pred_{}.pdf'.format('disease_roc_file.pdf', disease)

    # Plot disease specific PR
    plt.figure(figsize=(3, 3))
    aupr = []
    idx = 0
    for label, metrics in met_list:
        pr_df = metrics['pr_df']
        plt.plot(pr_df.recall, pr_df.precision,
                 label='{} (AUPR = {:.1%})'.format(label, metrics['aupr']),
                 linewidth=1, c=colors[idx])
        aupr.append(metrics['aupr'])
        idx += 1
    disease_aupr[disease] = aupr

    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=8)
    plt.ylabel('Precision', fontsize=8)
    plt.title('')
    plt.tick_params(labelsize=8)
    lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                     loc=2,
                     borderaxespad=0.,
                     fontsize=7.5)

    plt.savefig(disease_pr_sub_file, dpi=600, bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()

    # Plot disease specific ROC
    plt.figure(figsize=(3, 3))
    auroc = []
    idx = 0
    for label, metrics in met_list:
        roc_df = metrics['roc_df']
        plt.plot(roc_df.fpr, roc_df.tpr,
                 label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']),
                 linewidth=1, c=colors[idx])
        auroc.append(metrics['auroc'])
        idx += 1
    disease_auroc[disease] = auroc

    plt.axis('equal')
    plt.plot([0, 1], [0, 1], color='navy', linewidth=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.title('')
    plt.tick_params(labelsize=8)
    lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                     loc=2,
                     borderaxespad=0.,
                     fontsize=7.5)

    plt.savefig(disease_roc_sub_file, dpi=600, bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close()

index_lab = ['Train', 'Test', 'Cross Validation']

if shuffled:
    index_lab += ['Random']

disease_auroc_df = pd.DataFrame(disease_auroc, index=index_lab).T
disease_auroc_df = disease_auroc_df.sort_values('Cross Validation',
                                                ascending=False)
ax = disease_auroc_df.plot(kind='bar', title='Disease Specific Performance')
ax.set_ylabel('AUROC')
plt.tight_layout()
plt.savefig("dis_summary_auroc_file.pdf", dpi=600, bbox_inches='tight')
plt.close()

disease_aupr_df = pd.DataFrame(disease_aupr, index=index_lab).T
disease_aupr_df = disease_aupr_df.sort_values('Cross Validation',
                                              ascending=False)
ax = disease_aupr_df.plot(kind='bar', title='Disease Specific Performance')
ax.set_ylabel('AUPR')
plt.tight_layout()
plt.savefig("dis_summary_aupr_file.pdf", dpi=600, bbox_inches='tight')
plt.close()

# Save classifier coefficients
final_pipeline = cv_pipeline.best_estimator_
final_classifier = final_pipeline.named_steps['classify']

coef_df = pd.DataFrame.from_dict(
    {'feature': x_df.columns,
     'weight': final_classifier.coef_[0]})

coef_df['abs'] = coef_df['weight'].abs()
coef_df = coef_df.sort_values('abs', ascending=False)
coef_df.to_csv("classifier_file.csv", sep='\t')

if keep_inter:
    full_roc_df.to_csv("roc_results_file.csv", sep='\t')

# Apply the same classifier previously built to predict alternative genes
if alt_genes[0] is not 'None':
    # Classifying alternative mutations
    y_alt = mutation_df[alt_genes]

    # Add copy number info if applicable
    if copy_number:
        y_alt = integrate_copy_number(y=y_alt, cancer_genes_df=cancer_genes,
                                      genes=alt_genes, loss_df=copy_loss_df,
                                      gain_df=copy_gain_df)
    # Append disease id
    y_alt = y_alt.assign(total_status=y_alt.max(axis=1))
    y_alt = y_alt.reset_index().merge(sample_freeze,
                                      how='left').set_index('SAMPLE_BARCODE')

    # Filter data
    alt_count_df = y_alt.groupby('DISEASE').sum()
    alt_prop_df = alt_count_df.divide(y_alt['DISEASE'].value_counts(sort=False)
                                                      .sort_index(), axis=0)

    alt_count_table = alt_count_df.merge(alt_prop_df,
                                         left_index=True,
                                         right_index=True,
                                         suffixes=('_count', '_proportion'))
    alt_count_table.to_csv("alt_count_table_file.csv")

    mut_co = alt_count_df['total_status']
    prop = alt_prop_df['total_status']

    if alt_diseases[0] == 'Auto':
        alt_filter_dis = (mut_co > alt_filter_count) & (prop > alt_filter_prop)
        alt_diseases = alt_filter_dis.index[alt_filter_dis].tolist()

    # Subset data
    y_alt_df = y_alt[y_alt.DISEASE.isin(alt_diseases)].total_status
    common_alt_samples = list(set(y_alt_df.index) & set(rnaseq_full_df.index))

    y_alt_df = y_alt_df.loc[common_alt_samples]
    rnaseq_alt_df = rnaseq_full_df.loc[y_alt_df.index, :]

    y_alt_matrix = mut_burden.merge(pd.DataFrame(y_alt_df), right_index=True,
                                    left_on='SAMPLE_BARCODE')\
                             .set_index('SAMPLE_BARCODE')

    # Add Covariate Info to alternative y matrix
    y_alt_sub = y_alt.loc[y_alt_matrix.index]['DISEASE']
    covar_dummy_alt = pd.get_dummies(sample_freeze['DISEASE']).astype(int)
    covar_dummy_alt.index = sample_freeze['SAMPLE_BARCODE']
    covar_alt = covar_dummy_alt.merge(y_alt_matrix, right_index=True,
                                      left_index=True)
    covar_alt = covar_alt.drop('total_status', axis=1)
    y_alt_df = y_alt_df.loc[y_alt_sub.index]

    # Process alternative x matrix
    x_alt_df = rnaseq_alt_df.loc[y_alt_df.index, :]
    if x_matrix == 'raw':
        x_alt_df = x_alt_df.loc[:, mad_genes]

    x_alt_df_update = pd.DataFrame(fitted_scaler.transform(x_alt_df),
                                   columns=x_alt_df.columns)
    x_alt_df_update.index = x_alt_df.index
    x_alt_df = x_alt_df_update.merge(covar_alt, left_index=True,
                                     right_index=True)

    # Apply the previously fit model to predict the alternate Y matrix
    y_alt_cv = cv_pipeline.decision_function(X=x_alt_df)
    alt_metrics_cv = get_threshold_metrics(y_alt_df, y_alt_cv,
                                           drop_intermediate=keep_inter)

    validation_metrics = {}
    val_x_type = {}
    for disease in alt_diseases:
        sample_dis = y_alt_sub[y_alt_sub == disease].index

        # Subset full data if it has not been trained on
        if disease not in diseases:
            x_sub = x_alt_df.loc[sample_dis]
            y_sub = y_alt_df[sample_dis]
            category = 'Full'

        # Only subset to the holdout set if data was trained on
        else:
            x_sub = x_test.loc[x_test.index.isin(sample_dis)]
            y_sub = y_test[y_test.index.isin(sample_dis)]
            category = 'Holdout'

        # If there are not enough classes do not proceed to plot
        if y_sub.sum() < 1:
            continue

        neg, pos = y_sub.value_counts()
        val_x_type[disease] = [category, neg, pos]
        y_pred_alt = cv_pipeline.decision_function(x_sub)
        y_pred_alt_cv = y_alt_cv[y_alt_df.index.isin(y_sub.index)]

        alt_metrics_dis = get_threshold_metrics(y_sub, y_pred_alt,
                                                disease=disease,
                                                drop_intermediate=keep_inter)
        alt_metrics_di_cv = get_threshold_metrics(y_sub, y_pred_alt_cv,
                                                  disease=disease,
                                                  drop_intermediate=keep_inter)
        validation_metrics[disease] = [alt_metrics_dis, alt_metrics_di_cv]

    # Compile a summary dataframe
    val_x_type = pd.DataFrame.from_dict(val_x_type)
    val_x_type.index = ['class', 'negatives', 'positives']
    val_x_type.to_csv("alt_gene_summary_file.csv", sep='\t')

    alt_disease_auroc = {}
    alt_disease_aupr = {}
    for disease, metrics_val in validation_metrics.items():
        met_test, met_cv = metrics_val
        alt_disease_auroc[disease] = [met_test['auroc'], met_cv['auroc']]
        alt_disease_aupr[disease] = [met_test['aupr'], met_cv['aupr']]

    # Plot alternative gene cancer-type specific AUROC plots
    alt_disease_auroc_df = pd.DataFrame(alt_disease_auroc,
                                        index=['Hold Out', 'Full Data']).T
    alt_disease_auroc_df = alt_disease_auroc_df.sort_values('Full Data',
                                                            ascending=False)
    ax = alt_disease_auroc_df.plot(kind='bar', title='Alt Gene Performance')
    ax.set_ylim([0, 1])
    ax.set_ylabel('AUROC')
    plt.tight_layout()
    plt.savefig(alt_gene_auroc_file, dpi=600, bbox_inches='tight')
    plt.close()

    # Plot alternative gene cancer-type specific AUPR plots
    alt_disease_aupr_df = pd.DataFrame(alt_disease_aupr,
                                       index=['Hold Out', 'Full Data']).T
    alt_disease_aupr_df = alt_disease_aupr_df.sort_values('Full Data',
                                                          ascending=False)
    ax = alt_disease_aupr_df.plot(kind='bar', title='Alt Gene Performance')
    ax.set_ylim([0, 1])
    ax.set_ylabel('AUPR')
    plt.tight_layout()
    plt.savefig(alt_gene_aupr_file, dpi=600, bbox_inches='tight')
    plt.close()

# Write a summary for the inputs and outputs of the classifier
with open(os.path.join(base_folder, 'classifier_summary.txt'), 'w') as sum_fh:
    summarywriter = csv.writer(sum_fh, delimiter='\t')

    # Summarize parameters
    summarywriter.writerow(['Parameters:'])
    summarywriter.writerow(['Genes:'] + genes)
    summarywriter.writerow(['Diseases:'] + diseases)
    summarywriter.writerow(['Alternative Genes:'] + alt_genes)
    summarywriter.writerow(['Alternative Diseases:'] + alt_diseases)
    summarywriter.writerow(['Number of Features:', str(x_df.shape[1])])
    summarywriter.writerow(['Drop Gene:', drop])
    summarywriter.writerow(['Copy Number:', copy_number])
    summarywriter.writerow(['Alphas:'] + alphas)
    summarywriter.writerow(['L1_ratios:'] + l1_ratios)
    summarywriter.writerow(['Hypermutated Removed:', str(remove_hyper)])
    summarywriter.writerow([])

    # Summaryize results
    summarywriter.writerow(['Results:'])
    summarywriter.writerow(['Optimal Alpha:',
                            str(cv_pipeline.best_params_['classify__alpha'])])
    summarywriter.writerow(['Optimal L1:', str(cv_pipeline.best_params_
                                               ['classify__l1_ratio'])])
    summarywriter.writerow(['Coefficients:', classifier_file])
    summarywriter.writerow(['Training AUROC:', metrics_train['auroc']])
    summarywriter.writerow(['Testing AUROC:', metrics_test['auroc']])
    summarywriter.writerow(['Cross Validation AUROC', metrics_cv['auroc']])
    summarywriter.writerow(['Training AUPR:', metrics_train['aupr']])
    summarywriter.writerow(['Testing AUPR:', metrics_test['aupr']])
    summarywriter.writerow(['Cross Validation AUPR:', metrics_cv['aupr']])
    summarywriter.writerow(['Disease specific performance:'])
    for disease, auroc in disease_auroc.items():
        summarywriter.writerow(['', disease, 'Training AUROC:', auroc[0],
                                'Testing AUROC:', auroc[1],
                                'Cross Validation AUROC:', auroc[2]])
    for disease, aupr in disease_aupr.items():
        summarywriter.writerow(['', disease, 'Training AUPR:', aupr[0],
                                'Testing AUPR:', aupr[1],
                                'Cross Validation AUPR:', aupr[2]])
    if alt_genes[0] is not 'None':
        summarywriter.writerow(['Alternate gene performance:'] + alt_genes)
        summarywriter.writerow(['Alternative gene AUROC:',
                                str(alt_metrics_cv['auroc'])])
        summarywriter.writerow(['Alternative gene AUPR:',
                                str(alt_metrics_cv['aupr'])])
        for alt_dis, alt_auroc in alt_disease_auroc.items():
            summarywriter.writerow(['', alt_dis,
                                    'Holdout AUROC:', alt_auroc[0],
                                    'Full Data AUROC:', alt_auroc[1],
                                    'Category:', val_x_type[alt_dis]['class'],
                                    'num_positive:',
                                    str(val_x_type[alt_dis]['positives']),
                                    'num_negatives:',
                                    str(val_x_type[alt_dis]['negatives'])])
        for alt_dis, alt_aupr in alt_disease_aupr.items():
            summarywriter.writerow(['', alt_dis,
                                    'Holdout AUPR:', alt_aupr[0],
                                    'Full Data AUPR:', alt_aupr[1],
                                    'Category:', val_x_type[alt_dis]['class'],
                                    'num_positive:',
                                    str(val_x_type[alt_dis]['positives']),
                                    'num_negatives:',
                                    str(val_x_type[alt_dis]['negatives'])])


```

