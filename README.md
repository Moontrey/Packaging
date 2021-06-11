# Packaging
last task in ml eng course
# to install
!pip install last_task
# to use 
import last_task\
X, y, df = last_task.get_data('rawdata_new.csv')\
X = last_task.preproc(X)\
y_pred, clf = last_task.train_predict(X, y.to_numpy().ravel(), 'l2', 1, 'lbfgs')
