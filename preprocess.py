import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import *

def model_data():
    '''将 DataFrame 中的数据转换为可被模型直接使用的数据'''
    # 加载数据集并转换成 ndarray 类型
    df = get_data()
    df_array = df.to_numpy()

    # 特征向量与标签分离
    X = df_array[:, :-1]
    y = df_array[:, -1]

    # 分割出训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train_scaled, y_train, X_test_scaled, y_test

def load_data():
    """
    加载 OULAD 数据表并合并多表信息
    """
    # 读取各表
    student_info = pd.read_csv(os.path.join(Config.DATA_DIR, "studentInfo.csv"))
    student_registration = pd.read_csv(os.path.join(Config.DATA_DIR, "studentRegistration.csv"))
    student_vle = pd.read_csv(os.path.join(Config.DATA_DIR, "studentVle.csv"))
    student_assessment = pd.read_csv(os.path.join(Config.DATA_DIR, "studentAssessment.csv"))
    assessments = pd.read_csv(os.path.join(Config.DATA_DIR, "assessments.csv"))

    # 将 studentAssessment 表与 assessments 表通过 id_assessment 关联
    sa_full = pd.merge(student_assessment, assessments[["id_assessment", "code_module", "code_presentation"]],
                        on="id_assessment", how="left")

    # 统计每个学生-课程-学期的平均分和作业数
    sa_stats = sa_full.groupby(["id_student", "code_module", "code_presentation"]).agg(
        avg_score=("score", "mean"),
        num_assignments=("score", "count")
    ).reset_index()

    # 统计前n_weeks的VLE活跃度
    student_vle = student_vle.copy()
    student_vle["week"] = student_vle["date"] // 7
    vle_nw = student_vle[student_vle["week"] < Config.N_WEEKS]
    vle_stats = vle_nw.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique")
    ).reset_index()

    # 合并多个表
    df = pd.merge(student_info, student_registration[["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"]],
                    on=["id_student", "code_module", "code_presentation"], how="left")
    df = pd.merge(df, sa_stats, on=["id_student", "code_module", "code_presentation"], how="left")
    df = pd.merge(df, vle_stats, on=["id_student", "code_module", "code_presentation"], how="left")

    return df

def preprocess_data(df):
    """
    对合并后的数据进行清洗和独热编码，生成特征向量和标签。
    """
    # 去除缺失 final_result 的样本
    df = df.dropna(subset=["final_result"])
    # 生成标签：学业困难（Fail/Withdrawn）为1，其他为0
    df["label"] = df["final_result"].map({"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1})
    # 选择特征
    feature_cols = [
        "age_band", "gender", "imd_band", "disability", "num_of_prev_attempts", "studied_credits",
        "avg_score", "num_assignments", "total_clicks", "active_days"
    ]
    df = df.dropna(subset=feature_cols + ["label"])
    # 类别特征独热编码
    df = pd.get_dummies(df[feature_cols + ["label"]], columns=["age_band", "gender", "imd_band", "disability"])
    # 特征标准化
    feature_cols_all = [c for c in df.columns if c != 'label']
    scaler = StandardScaler()
    X = df[feature_cols_all]
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols_all, index=df.index)
    df_scaled['label'] = df['label'].values
    return df_scaled

def get_data():
    df = load_data()
    prep_df = preprocess_data(df)
    return prep_df



if __name__ == '__main__':
    df = get_data()
    df_array = df.to_numpy()
    print(df, type(df))
    print(df_array, type(df_array))
