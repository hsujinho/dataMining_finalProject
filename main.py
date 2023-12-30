import numpy as np
import pandas as pd
import time
from myDecTreeReg import decisionTreeRegressor

def rearrange_dataset(train_data, test_data):
    # 1.train
    skills_list = []
    for i in range(len(train_data)):
        zip_list = train_data['Skills'][i].replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(
            ',')
        dict_temp = {}
        str_ = ''
        for j in range(len(zip_list)):
            dict_temp[zip_list[j]] = 1
            str_ += zip_list[j] + " "
        dict_temp['skills_clean'] = str_
        skills_list.append(dict_temp)
    train_data = pd.concat([train_data, pd.DataFrame(skills_list).fillna(0)], axis=1)
    # 2.test
    skills_list = []
    for i in range(len(test_data)):
        zip_list = test_data['Skills'][i].replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(',')
        dict_temp = {}
        str_ = ''
        for j in range(len(zip_list)):
            dict_temp[zip_list[j]] = 1
            str_ += zip_list[j] + " "
        dict_temp['skills_clean'] = str_
        skills_list.append(dict_temp)

    test_data = pd.concat([test_data, pd.DataFrame(skills_list).fillna(0)], axis=1)
    # 1.train
    job_group_list = []
    # ['Jobs_Group']
    for i in range(len(train_data)):
        zip_list = train_data['Jobs_Group'][i]
        dict_temp = {}
        dict_temp[zip_list] = 1
        job_group_list.append(dict_temp)
    train_data = pd.concat([train_data, pd.DataFrame(job_group_list).fillna(0)], axis=1)

    # 2.test
    job_group_list = []
    for i in range(len(test_data)):
        zip_list = test_data['Jobs_Group'][i]
        dict_temp = {}
        dict_temp[zip_list] = 1
        job_group_list.append(dict_temp)
    test_data = pd.concat([test_data, pd.DataFrame(job_group_list).fillna(0)], axis=1)

    return train_data, test_data

trainDataFrame = pd.read_csv('usjobs_train.csv')
testDataFrame = pd.read_csv('usjobs_test.csv')

train_data1, test_data1 = rearrange_dataset(train_data=trainDataFrame, test_data=testDataFrame)

train_data1['Director_Score'].fillna(np.mean(train_data1['Director_Score']), inplace=True)
train_data1['Company_Score'].fillna(np.mean(train_data1['Company_Score']), inplace=True)
train_data1['Reviews'].fillna(np.mean(train_data1['Reviews']), inplace=True)

test_data1['Director_Score'].fillna(np.mean(test_data1['Director_Score']), inplace=True)
test_data1['Company_Score'].fillna(np.mean(test_data1['Company_Score']), inplace=True)
test_data1['Reviews'].fillna(np.mean(test_data1['Reviews']), inplace=True)

train_data_revenue = []
for i in range(len(train_data1)):
    zip_list = train_data1['Revenue'][i]
    # if nan value, skip
    if type(zip_list) == float:
        continue
    if zip_list not in train_data_revenue:
        train_data_revenue.append(zip_list)

test_data_revenue = []
for i in range(len(test_data1)):
    zip_list = test_data1['Revenue'][i]
    # if nan value, skip
    if type(zip_list) == float:
        continue
    if zip_list not in test_data_revenue:
        test_data_revenue.append(zip_list)

size_mapping = {'XXXS': 1, 'XXS': 2, 'XS': 3, 'S': 4, 'M': 5, 'L': 6, 'XL': 7, 'XXL': 8, 'XXXL': 9}
# 將 size_mapping 的值轉換為數值
size_mapping = {key: int(value) for key, value in size_mapping.items()}

# 將原本的資料對應到 size_mapping 的新值，沒有對應到的值不做轉換
train_data1['Revenue'] = train_data1['Revenue'].map(size_mapping)
train_data1['Revenue'].fillna(np.mean(train_data1['Revenue']), inplace=True)
test_data1['Revenue'] = test_data1['Revenue'].map(size_mapping)
test_data1['Revenue'].fillna(np.mean(test_data1['Revenue']), inplace=True)
train_data1['Employee'] = train_data1['Employee'].map(size_mapping)
train_data1['Employee'].fillna(np.mean(train_data1['Employee']), inplace=True)
test_data1['Employee'] = test_data1['Employee'].map(size_mapping)
test_data1['Employee'].fillna(np.mean(test_data1['Employee']), inplace=True)

# training data with decision tree regressor

startTime = time.time()

model = decisionTreeRegressor(max_depth=2)
X = train_data1[
    ['Revenue', 'Employee', 'Company_Score', 'Director_Score', 'Reviews', 'ArtificialIntelligence', 'MachineLearning',
     'Python', 'TensorFlow',
     'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master', 'Office', 'Docker', 'Word', 'Azure',
     'Hadoop', 'Airflow', 'MBA', 'CPA',
     'Snowflake', 'Databricks', 'SQL', '', 'GoogleCloud', 'SciKit', 'Pandas', 'ChatGPT', 'Agile', 'R', 'NeuralNetwork',
     'Jupyter', 'PowerPoint', 'Spanish',
     'NumPy', 'React', 'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#', 'Rust', 'VBA',
     'Bachelor', 'Analyst', 'Business Analyst',
     'Business Intelligence', 'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Finance',
     'Financial Analyst', 'ML/AI Engineer',
     'Operations Analyst', 'Others', 'Statistician/Mathemathics']]
y = train_data1['Mean_Salary']

model.fit(X, y)

trainTime = time.time() - startTime

def train_test_split(X, y, ratio):
    x_df = X
    y_df = y
    total_rows = x_df.shape[0]
    train_size = int(total_rows * (1 - ratio))
    x_train1 = x_df[0:train_size]
    y_train1 = y_df[0:train_size].values
    x_test1 = x_df[train_size:]
    y_test1 = y_df[train_size:].values

    # split all categorical and Mean_Salary
    # x -> categorical
    # y -> Mean_Salary
    return x_train1, x_test1, y_train1, y_test1


def calculate_accuracy(predicted_labels, true_labels):
    # count max_error and mean_error
    max_error = 0
    mean_square_error = 0
    for i in range(len(predicted_labels)):
        error = abs(predicted_labels[i] - true_labels[i])
        mean_square_error += error ** 2
        if error > max_error:
            max_error = error
    mean_square_error /= len(predicted_labels)
    print("max_error:", max_error)
    print("mean_square_error:", mean_square_error)


X_train, X_test, y_train, y_test = train_test_split(
    train_data1[
        ['Revenue', 'Employee', 'Company_Score', 'Director_Score', 'Reviews', 'ArtificialIntelligence',
         'MachineLearning',
         'Python', 'TensorFlow',
         'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master', 'Office', 'Docker', 'Word', 'Azure',
         'Hadoop', 'Airflow', 'MBA', 'CPA',
         'Snowflake', 'Databricks', 'SQL', '', 'GoogleCloud', 'SciKit', 'Pandas', 'ChatGPT', 'Agile', 'R',
         'NeuralNetwork',
         'Jupyter', 'PowerPoint', 'Spanish',
         'NumPy', 'React', 'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#', 'Rust', 'VBA',
         'Bachelor', 'Analyst', 'Business Analyst',
         'Business Intelligence', 'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Finance',
         'Financial Analyst', 'ML/AI Engineer',
         'Operations Analyst', 'Others', 'Statistician/Mathemathics']],
    train_data1['Mean_Salary'],
    ratio=0.2)

calculate_accuracy(model.predict(X_test), y_train)

# # testing data with decision tree regressor

startTime = time.time()

X_test = test_data1[
    ['Revenue', 'Employee', 'Company_Score', 'Director_Score', 'Reviews', 'ArtificialIntelligence', 'MachineLearning',
     'Python', 'TensorFlow',
     'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master', 'Office', 'Docker', 'Word', 'Azure',
     'Hadoop', 'Airflow', 'MBA', 'CPA',
     'Snowflake', 'Databricks', 'SQL', '', 'GoogleCloud', 'SciKit', 'Pandas', 'ChatGPT', 'Agile', 'R', 'NeuralNetwork',
     'Jupyter', 'PowerPoint', 'Spanish',
     'NumPy', 'React', 'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#', 'Rust', 'VBA',
     'Bachelor', 'Analyst', 'Business Analyst',
     'Business Intelligence', 'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Finance',
     'Financial Analyst', 'ML/AI Engineer',
     'Operations Analyst', 'Others', 'Statistician/Mathemathics']]

y_pred = model.predict(X_test)

predictTime = time.time() - startTime

print("Depth: ", model.max_depth)
print("Training time: ", trainTime)
print("Predicting time: ", predictTime)
print("Total time: ", trainTime + predictTime)

# output result to csv file with column name: jobID, Mean_Salary

outputDF = pd.DataFrame({'ID': testDataFrame['ID'], 'Mean_Salary': y_pred})
fileName = 'output' + f"{model.max_depth}" + '.csv'
outputDF.to_csv(fileName, index=False)