#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyodbc pandas pypyodbc')


# In[2]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Fetch table names
tables = cursor.tables().fetchall()
print("Tables in the database:")
for i, table in enumerate(tables):
    print(i)
    print(table.table_name)

# Example: Fetch all rows from a specific table
table_name = 'injury'  # Replace with your table name
query = f'SELECT * FROM {table_name}'

# Execute the query and fetch the data into a pandas DataFrame
data = pd.read_sql(query, connection)

# Display the data
print(data)

# Close the connection
connection.close()


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[9]:


data['inj_person_category'] = encoder.fit_transform(data['inj_person_category'])


# In[10]:


data.head()


# In[11]:


data


# In[12]:


data['injury_level'] = encoder.fit_transform(data['injury_level'])


# In[13]:


data


# In[14]:


data['lchg_userid'] = encoder.fit_transform(data['lchg_userid'])


# In[15]:


data


# In[16]:


data['inj_person_category'].value_counts()


# In[17]:


data['injury_level'].value_counts()


# In[18]:


data['inj_person_count'].value_counts()


# In[19]:


data['lchg_userid'].value_counts()


# In[20]:


injury_level_counts = data['injury_level'].value_counts()


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


plt.figure(figsize=(8, 8))
plt.pie(injury_level_counts, labels=injury_level_counts.index, autopct='%1.2f%%', startangle=140)
plt.title('Distribution of Injury Person Category')
plt.show()


# In[23]:


injured_persons = data['inj_person_count'].value_counts()


# In[24]:


plt.figure(figsize=(8, 8))
plt.pie(injured_persons, labels=injured_persons.index, autopct='%1.2f%%', startangle=140)
plt.title('Distribution of Injury Person Counts')
plt.show()


# In[25]:


injured_category = data['inj_person_category'].value_counts()


# In[26]:


plt.figure(figsize=(8, 8))
plt.pie(injured_category, labels=injured_category.index, autopct='%1.0f%%', startangle=100)
plt.title('Distribution of Injury Person Category')
plt.legend()
plt.show()


# In[27]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Example: Fetch all rows from a specific table
table_name = 'dt_aircraft'  # Replace with your table name
query = f'SELECT * FROM {table_name}'

# Execute the query and fetch the data into a pandas DataFrame
data1 = pd.read_sql(query, connection)

# Display the data
print(data1)

# Close the connection
connection.close()


# In[28]:


data1.isnull().sum()


# In[29]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Example: Fetch all rows from a specific table
table_name = 'dt_events'  # Replace with your table name
query = f'SELECT * FROM {table_name}'

# Execute the query and fetch the data into a pandas DataFrame
data2 = pd.read_sql(query, connection)

# Display the data
print(data2)

# Close the connection
connection.close()


# In[30]:


data2.isnull().sum()


# In[31]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Example: Fetch all rows from a specific table
table_name = 'events'  # Replace with your table name
query = f'SELECT * FROM {table_name}'

# Execute the query and fetch the data into a pandas DataFrame
data3 = pd.read_sql(query, connection)

# Display the data
print(data3)

# Close the connection
connection.close()


# In[32]:


data3.isnull().sum()


# In[33]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Example: Fetch all rows from a specific table
table_name = 'Findings'  # Replace with your table name
query = f'SELECT * FROM {table_name}'

# Execute the query and fetch the data into a pandas DataFrame
data4 = pd.read_sql(query, connection)

# Display the data
print(data4)

# Close the connection
connection.close()


# In[34]:


data4.isnull().sum()


# In[35]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

# Load tables into pandas DataFrames
injury = pd.read_sql('SELECT * FROM injury', connection)                            
dt_aircraft = pd.read_sql('SELECT * FROM dt_aircraft', connection)
dt_events = pd.read_sql('SELECT * FROM dt_events', connection)
events = pd.read_sql('SELECT * FROM events', connection)
Findings = pd.read_sql('SELECT * FROM Findings', connection)

# Drop 'lchg_userid' column from all tables
tables = [injury, dt_aircraft, dt_events, events, Findings]
for table in tables:
    table.drop(columns=['lchg_userid'], inplace=True, errors='ignore')


# In[36]:


import pyodbc
import pandas as pd
import numpy as np

# Path to your .mdb file
mdb_file_path = r'E:\datasets\avall\avall.mdb'

# Connection string to connect to the Access database
connection_string = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={mdb_file_path};'
)

# Establish the connection
connection = pyodbc.connect(connection_string)

# Create a cursor object
cursor = connection.cursor()

def load_table(table_name):
    query = f'SELECT * FROM {table_name}'
    return pd.read_sql(query, connection)

# Load tables into pandas DataFrames
tables = {}
table_names = [
    'injury', 'dt_aircraft', 'dt_events', 'events', 
     'Findings'
]

for name in table_names:
    tables[name] = load_table(name)

tables = [injury, events, Findings, dt_aircraft, dt_events]

# Handle missing values
tables[0].fillna({'inj_person_count': 0}, inplace=True)
tables[1].fillna({'wx_cond_basic': 'unknown', 'faa_dist_office': 'unknown', 'dec_latitude': 0, 'dec_longitude': 0}, inplace=True)
tables[2].fillna({'Cause_Factor': 'unknown'}, inplace=True)


# In[37]:


tables[0].isnull().sum()


# In[38]:


tables[0].shape


# In[39]:


tables[1].isnull().sum()


# In[40]:


tables[1].shape


# In[41]:


tables[2].isnull().sum()


# In[42]:


tables[2].shape


# In[43]:


tables[3].isnull().sum()


# In[44]:


tables[3].shape


# In[45]:


tables[4].isnull().sum()


# In[46]:


tables[4].shape


# In[47]:


import pandas as pd

tables = [injury, events, Findings, dt_aircraft, dt_events]

# Sample a subset of each table (e.g., 30% of the data)
sampled_tables = [table.sample(frac=0.3, random_state=42) for table in tables]

# Check the shape of the sampled tables
for i, table in enumerate(sampled_tables):
    print(f"Sampled Table {i} shape: {table.shape}")

# Initialize merged data with the first sampled table
merged_data = sampled_tables[0]

# Merge tables one by one
for i in range(1, len(sampled_tables)):
    suffix = f'_t{i}'
    merged_data = pd.merge(merged_data, sampled_tables[i], on='ev_id', how='inner', suffixes=('', suffix))
    print(f"After merging table {i}, shape: {merged_data.shape}")
    print(merged_data.head())

# Print final merged data shape
print(f"Final merged data shape: {merged_data.shape}")


# In[48]:


columns_to_drop = [
    'wx_brief_comp', 'wx_dens_alt', 'wx_int_precip', 'metar',
    'vis_rvr', 'vis_rvv', 'wind_vel_kts', 'sky_cond_ceil', 
    'wx_src_iic', 'wx_obs_time', 'wx_obs_fac_id', 'wx_obs_elev', 
    'wx_obs_dist', 'wx_obs_tmzn', 'ntsb_docket', 'ntsb_notf_from',
    'ntsb_notf_date', 'ntsb_notf_tm', 'fiche_number', 'apt_name',
    'ev_nr_apt_id', 'apt_dir', 'apt_elev', 'vis_sm', 'sky_cond_nonceil',
    'sky_nonceil_ht', 'wind_dir_ind', 'wind_vel_ind', 'gust_ind',
    'gust_kts', 'altimeter', 'lchg_date_t1', 'wx_cond_basic', 'faa_dist_office',
    'dec_latitude', 'dec_longitude', 'Aircraft_Key_t2', 'finding_no',
    'finding_code', 'finding_description', 'category_no', 'subcategory_no',
    'section_no', 'subsection_no', 'modifier_no', 'Cause_Factor', 'lchg_date_t2',
    'cm_inPc', 'Aircraft_Key_t3', 'col_name', 'code', 'lchg_date_t3',
    'col_name_t4', 'code_t4', 'lchg_date_t4'
]

# Drop the columns from the DataFrame
merged_data = merged_data.drop(columns=columns_to_drop)


# In[49]:


merged_data


# In[50]:


merged_data.shape


# In[51]:


merged_data.columns.tolist()


# In[52]:


merged_data.isnull().sum().tolist()


# In[53]:


import pandas as pd
# Assuming merged_data is your final DataFrame after merging the tables
# Fill missing values with appropriate strategies

fill_values = {
    'ev_time': 'unknown',
    'ev_tmzn': 'unknown',
    'ev_city': 'unknown',
    'ev_state': 'unknown',
    'ev_site_zipcode': 'unknown',
    'mid_air': 0,
    'on_ground_collision': 0,
    'latitude': 0,
    'longitude': 0,
    'latlong_acq': 'unknown',
    'apt_name': 'unknown',
    'ev_nr_apt_id': 'unknown',
    'ev_nr_apt_loc': 'unknown',
    'apt_dist': 0,
    'apt_dir': 'unknown',
    'wx_brief_comp': 'unknown',
    'wx_src_iic': 'unknown',
    'wx_obs_time': 'unknown',
    'wx_obs_dir': 'unknown',
    'wx_obs_fac_id': 'unknown',
    'wx_obs_elev': 0,
    'wx_obs_dist': 0,
    'wx_obs_tmzn': 'unknown',
    'sky_cond_nonceil': 'unknown',
    'sky_nonceil_ht': 0,
    'sky_ceil_ht': 0,
    'sky_cond_ceil': 'unknown',
    'vis_rvr': 0,
    'vis_rvv': 0,
    'vis_sm': 0,
    'wx_temp': 0,
    'wx_dew_pt': 0,
    'wind_dir_deg': 0,
    'wind_dir_ind': 'unknown',
    'gust_ind': 0,
    'gust_kts': 0,
    'altimeter': 0,
    'wx_dens_alt': 0,
    'wx_int_precip': 'unknown',
    'metar': 'unknown',
    'wx_cond_basic': 'unknown',
    'faa_dist_office': 'unknown',
    'dec_latitude': 0,
    'dec_longitude': 0
}

for column, value in fill_values.items():
    if column in merged_data.columns:
        merged_data[column].fillna(value, inplace=True)

# Display the final merged DataFrame
print(merged_data.head())


# In[54]:


merged_data.isnull().sum().tolist()


# In[55]:


# Assuming merged_data is your DataFrame

# Count null values in each column
null_counts = merged_data.isnull().sum().tolist()
columns = merged_data.columns.tolist()

# Zip columns and their corresponding null counts
column_null_pairs = list(zip(columns, null_counts))

# Alternatively, create a DataFrame to display the pairs nicely
null_info_df = pd.DataFrame(column_null_pairs, columns=['Column', 'Null Count'])
print(null_info_df)


# In[56]:


# Filling missing values

# Fill categorical columns with mode
categorical_cols_mode = ['ev_time', 'ev_tmzn', 'ev_state', 'ev_site_zipcode', 'latlong_acq', 'ev_nr_apt_loc', 'light_cond', 'invest_agy']
for col in categorical_cols_mode:
    merged_data[col].fillna(merged_data[col].mode()[0], inplace=True)

# Fill binary columns with zero
binary_cols_zero = ['mid_air', 'on_ground_collision', 'inj_f_grnd', 'inj_m_grnd', 'inj_s_grnd']
for col in binary_cols_zero:
    merged_data[col].fillna(0, inplace=True)

# Fill numerical columns with median
numerical_cols_median = ['wx_obs_dir', 'sky_ceil_ht', 'wx_temp', 'wx_dew_pt', 'wind_dir_deg']
for col in numerical_cols_median:
    merged_data[col].fillna(merged_data[col].median(), inplace=True)


# In[57]:


merged_data.isnull().sum().tolist()


# In[58]:


import pandas as pd

# Assuming 'data' is your DataFrame

# Separate columns by data types
no_of_cc = 0
no_of_nc = 0
categorical_columns = merged_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = merged_data.select_dtypes(include=['number']).columns.tolist()

for i in categorical_columns:
    no_of_cc += 1
for i in numerical_columns:
    no_of_nc += 1

print(no_of_cc)
print(no_of_nc)
print("Categorical Columns:")
print(categorical_columns)
print("\nNumerical Columns:")
print(numerical_columns)


# In[59]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# List of categorical columns
categorical_columns = [
    'ev_id', 'inj_person_category', 'injury_level', 'ntsb_no', 'ev_type', 'ev_dow', 'ev_tmzn', 'ev_city', 'ev_state', 'ev_country', 'ev_site_zipcode', 'mid_air', 'on_ground_collision', 'latitude', 'longitude', 'latlong_acq', 'ev_nr_apt_loc', 'light_cond', 'ev_highest_injury', 'invest_agy'
]

# Initialize LabelEncoder
le = LabelEncoder()

# Encode each categorical column
for col in categorical_columns:
    merged_data[col] = le.fit_transform(merged_data[col].astype(str))

# Now 'data' DataFrame has encoded categorical columns
print(merged_data.head())  # To check the first few rows of the encoded DataFrame


# In[60]:


import pandas as pd

# Assuming 'data' is your DataFrame

# Separate columns by data types
no_of_cc = 0
no_of_nc = 0
categorical_columns = merged_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = merged_data.select_dtypes(include=['number']).columns.tolist()

for i in categorical_columns:
    no_of_cc += 1
for i in numerical_columns:
    no_of_nc += 1

print(no_of_cc)
print(no_of_nc)
print("Categorical Columns:")
print(categorical_columns)
print("\nNumerical Columns:")
print(numerical_columns)


# In[61]:


datetime_columns = merged_data.select_dtypes(include=['datetime64']).columns
datetime_columns


# In[62]:


for column in merged_data.columns:
    print(f"{column}: {merged_data[column].dtype}")


# In[63]:


merged_data.isnull().sum().tolist()


# In[64]:


# Extract year, month, and day from 'lchg_date' and 'ev_date'
merged_data['lchg_date_year'] = merged_data['lchg_date'].dt.year
merged_data['lchg_date_month'] = merged_data['lchg_date'].dt.month
merged_data['lchg_date_day'] = merged_data['lchg_date'].dt.day

merged_data['ev_date_year'] = merged_data['ev_date'].dt.year
merged_data['ev_date_month'] = merged_data['ev_date'].dt.month
merged_data['ev_date_day'] = merged_data['ev_date'].dt.day

# Drop the original datetime columns if no longer needed
merged_data = merged_data.drop(columns=['lchg_date', 'ev_date'])

# Check the changes
print(merged_data.head())


# In[65]:


for column in merged_data.columns:
    print(f"{column}: {merged_data[column].dtype}")


# In[66]:


import pandas as pd
import numpy as np

# Replace 'UNKNOWN' with NaN
merged_data.replace('UNKNOWN', np.nan, inplace=True)

# Identify and handle non-numeric columns
non_numeric_columns = merged_data.select_dtypes(include=['object']).columns

# Print non-numeric columns to verify
print("Non-numeric columns:", non_numeric_columns)

# Convert 'ev_time' to a numerical format if necessary (e.g., extract hour and minute)
# This is just an example, adjust based on actual format
merged_data['ev_time'] = pd.to_datetime(merged_data['ev_time'], errors='coerce').dt.hour

# Fill or drop NaN values (example: fill with median, mean, or a constant)
# Here we use mean for simplicity, but choose based on your needs
for column in non_numeric_columns:
    if merged_data[column].dtype == 'object':
        merged_data[column].fillna(merged_data[column].mode()[0], inplace=True)  # Fill with mode (most frequent value)

# Check for any remaining non-numeric columns
remaining_non_numeric = merged_data.select_dtypes(include=['object']).columns
print("Remaining non-numeric columns after handling:", remaining_non_numeric)

# Convert all columns to numeric if possible
merged_data = merged_data.apply(pd.to_numeric, errors='coerce')

# Fill remaining NaN values with mean
merged_data.fillna(merged_data.mean(), inplace=True)

# Verify changes
print(merged_data.dtypes)
print(merged_data.head())


# In[67]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[68]:


# Specify the target column
target_column = 'ev_highest_injury'


# In[69]:


# Split the data into features (X) and target (y)
X = merged_data.drop(columns=[target_column])
y = merged_data[target_column]


# In[70]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


# Create and train the RandomForestClassifier using "entropy"
rfc = RandomForestClassifier(criterion='entropy', 
                             n_estimators=100, 
                             max_depth=10, 
                             random_state=101)


# In[72]:


rfc.fit(X_train, y_train)


# In[73]:


# Predict on the test set
y_pred = rfc.predict(X_test)


# In[74]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[75]:


print("Random Forest with Entropy")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[76]:


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rfc, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())