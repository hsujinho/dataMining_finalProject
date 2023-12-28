import numpy as np
import pandas as pd
import time

# implementation of decision tree regressor
class node: 
    def __init__(self, value, left, right, attribute, threshold):
        self.value = value          # mean of y values
        self.left = left            # left subtree
        self.right = right          # right subtree
        self.attribute = attribute  # attribute to split on
        self.threshold = threshold  # threshold value to split on
# decision tree regressor class
class decisionTreeRegressor:
    def __init__(self, max_depth = 3):
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, y):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        self.tree = self.build_tree(X_np, y_np, depth=0)
    # find best split for a node
    def find_best_split(self, X, y):
        m, n = X.shape
        # if only one data point, return None
        if m <= 1:
            return None, None
        
        best_attribute = None
        best_threshold = None
        best_loss = np.inf
        # iterate through all attributes
        for att in range(n):
            # iterate through all possible thresholds
            thresholds = np.unique(X[:, att])
            for threshold in thresholds:
                # split data
                left = X[:, att] <= threshold
                right = X[:, att] > threshold
                y_left = y[left]
                y_right = y[right]
                # if no split, skip
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # calculate mean squared error
                MSE_left = np.mean((y[left] - np.mean(y[left]))**2)
                MSE_right = np.mean((y[right] - np.mean(y[right]))**2)
                # update best split if total MSE is lower
                if MSE_left + MSE_right < best_loss:
                    best_attribute = att
                    best_threshold = threshold
                    best_loss = MSE_left + MSE_right
        
        return best_attribute, best_threshold 
    # build tree recursively
    def build_tree(self, X, y, depth):
        #find best split
        best_attribute, best_threshold = self.find_best_split(X, y)
        #if no split or max depth reached, return leaf node
        if best_attribute is None or depth == self.max_depth:
            leaf_value = np.mean(y)
            return node(leaf_value, None, None, best_attribute, best_threshold)
        #if split exists, build tree recursively
        left = X[:, best_attribute] <= best_threshold
        right = X[:, best_attribute] > best_threshold
        left_child = self.build_tree(X[left], y[left], depth+1)
        right_child = self.build_tree(X[right], y[right], depth+1)
        return node(np.mean(y), left_child, right_child, best_attribute, best_threshold)
    # predict instance recursively
    def predict_instance(self, x, tree):
        #if is leaf node, return value
        if tree.left is None and tree.right is None:
            return tree.value
        #if it is not leaf node, compare attribute value to threshold and go left or right
        if x[tree.attribute] <= tree.threshold:
            return self.predict_instance(x, tree.left)
        return self.predict_instance(x, tree.right)
    # predict dataset
    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        predictions = []
        for x in X_np:
            predictions.append(self.predict_instance(x, self.tree))
        return np.array(predictions)
                

def rearrange_dataset(train_data, test_data):
    #1.train 
    skills_list = []
    for i in range(len(train_data)):
        zip_list = train_data['Skills'][i].replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
        dict_temp = {}
        str_ = ''
        for j in range(len(zip_list)):
            dict_temp[zip_list[j]] = 1
            str_ += zip_list[j] + " "
        dict_temp['skills_clean'] = str_
        skills_list.append(dict_temp)
    train_data = pd.concat([train_data,pd.DataFrame(skills_list).fillna(0)],axis =1)  
    #2.test
    skills_list = []
    for i in range(len(test_data)):
        zip_list = test_data['Skills'][i].replace('[','').replace(']','').replace("'",'').replace(" ",'').split(',')
        dict_temp = {}
        str_ = ''
        for j in range(len(zip_list)):
            dict_temp[zip_list[j]] = 1
            str_ += zip_list[j] + " "
        dict_temp['skills_clean'] = str_        
        skills_list.append(dict_temp)
        
    test_data = pd.concat([test_data,pd.DataFrame(skills_list).fillna(0)],axis =1)      
    #1.train 
    job_group_list = []
    # ['Jobs_Group']
    for i in range(len(train_data)):
        zip_list = train_data['Jobs_Group'][i]
        dict_temp = {}
        dict_temp[zip_list] = 1
        job_group_list.append(dict_temp)
    train_data = pd.concat([train_data,pd.DataFrame(job_group_list).fillna(0)],axis =1)
    
    #2.test
    job_group_list = []
    for i in range(len(test_data)):
        zip_list = test_data['Jobs_Group'][i]
        dict_temp = {}
        dict_temp[zip_list] = 1
        job_group_list.append(dict_temp)
    test_data = pd.concat([test_data,pd.DataFrame(job_group_list).fillna(0)],axis =1)
        
    
    return train_data, test_data

trainDataFrame = pd.read_csv('usjobs_train.csv')
testDataFrame = pd.read_csv('usjobs_test.csv')

train_data1, test_data1 = rearrange_dataset(train_data= trainDataFrame, test_data= testDataFrame)
train_data1['Director_Score'].fillna(np.mean(train_data1['Director_Score']), inplace= True)
train_data1['Company_Score'].fillna(np.mean(train_data1['Company_Score']), inplace= True)
train_data1['Reviews'].fillna(np.mean(train_data1['Reviews']), inplace= True)
train_data1[['Job','Company','skills_clean','ArtificialIntelligence', 'MachineLearning', 'Python', 'TensorFlow',
           'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master',
           'Office', 'Docker', 'Word', 'Azure', 'Hadoop', 'Airflow', 'MBA', 'CPA',
           'Director_Score', 'Snowflake', 'Databricks', 'SQL', '', 'Company_Score',
           'GoogleCloud', 'SciKit', 'Reviews', 'Pandas', 'ChatGPT', 'Agile', 'R',
           'NeuralNetwork', 'Jupyter', 'PowerPoint', 'Spanish', 'NumPy', 'React',
           'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#',
           'Rust', 'VBA', 'Bachelor', 'Analyst', 'Business Analyst', 'Business Intelligence', 
           'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 
           'Data Scientist', 'Finance', 'Financial Analyst', 
           'ML/AI Engineer', 'Operations Analyst', 'Others', 
           'Statistician/Mathemathics']].isna().sum()

test_data1['Director_Score'].fillna(np.mean(test_data1['Director_Score']), inplace= True)
test_data1['Company_Score'].fillna(np.mean(test_data1['Company_Score']), inplace= True)
test_data1['Reviews'].fillna(np.mean(test_data1['Reviews']), inplace= True)


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

# print(train_data_revenue)
size_mapping = {'XXXS': 1, 'XXS': 2, 'XS': 3, 'S': 4, 'M': 5, 'L': 6, 'XL': 7, 'XXL': 8, 'XXXL': 9}

# 將 size_mapping 的值轉換為數值
size_mapping = {key: int(value) for key, value in size_mapping.items()}

# print(size_mapping)
# 將原本的資料對應到 size_mapping 的新值，沒有對應到的值不做轉換
train_data1['Revenue'] = train_data1['Revenue'].map(size_mapping)
train_data1['Revenue'].fillna(np.mean(train_data1['Revenue']), inplace= True)
test_data1['Revenue'] = test_data1['Revenue'].map(size_mapping)
test_data1['Revenue'].fillna(np.mean(test_data1['Revenue']), inplace= True)
train_data1['Employee'] = train_data1['Employee'].map(size_mapping)
train_data1['Employee'].fillna(np.mean(train_data1['Employee']), inplace= True)
test_data1['Employee'] = test_data1['Employee'].map(size_mapping)
test_data1['Employee'].fillna(np.mean(test_data1['Employee']), inplace= True)

# training data with decision tree regressor

startTime = time.time()

model = decisionTreeRegressor(max_depth = 5)
X = train_data1[['Revenue', 'Employee', 'Company_Score', 'Director_Score', 'Reviews', 'ArtificialIntelligence', 'MachineLearning', 'Python', 'TensorFlow', 
                'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master', 'Office', 'Docker', 'Word', 'Azure', 'Hadoop', 'Airflow', 'MBA', 'CPA', 
                'Snowflake', 'Databricks', 'SQL', '', 'GoogleCloud', 'SciKit', 'Pandas', 'ChatGPT', 'Agile', 'R', 'NeuralNetwork', 'Jupyter', 'PowerPoint', 'Spanish', 
                'NumPy', 'React', 'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#', 'Rust', 'VBA', 'Bachelor', 'Analyst', 'Business Analyst', 
                'Business Intelligence', 'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Finance', 'Financial Analyst', 'ML/AI Engineer', 
                'Operations Analyst', 'Others', 'Statistician/Mathemathics']]
y = train_data1['Mean_Salary']

model.fit(X, y)

trainTime = time.time() - startTime

startTime = time.time()

# # testing data with decision tree regressor

X_test = test_data1[['Revenue', 'Employee', 'Company_Score', 'Director_Score', 'Reviews', 'ArtificialIntelligence', 'MachineLearning', 'Python', 'TensorFlow',
                    'Excel', 'Spark', 'PhD', 'AWS', 'C++', 'DeepLearning', 'Java', 'Master', 'Office', 'Docker', 'Word', 'Azure', 'Hadoop', 'Airflow', 'MBA', 'CPA',
                    'Snowflake', 'Databricks', 'SQL', '', 'GoogleCloud', 'SciKit', 'Pandas', 'ChatGPT', 'Agile', 'R', 'NeuralNetwork', 'Jupyter', 'PowerPoint', 'Spanish',
                    'NumPy', 'React', 'GitHub', 'Looker', 'English', 'Access', 'Matplotlib', 'Seaborn', 'C#', 'Rust', 'VBA', 'Bachelor', 'Analyst', 'Business Analyst',
                    'Business Intelligence', 'CFO', 'Controller', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Finance', 'Financial Analyst', 'ML/AI Engineer',
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
