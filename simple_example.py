from solnml.estimators import Classifier


clf = Classifier(dataset_name='iris',
                 time_limit=150,
                 output_dir='logs/',
                 ensemble_method='stacking',
                 evaluation='holdout',
                 metric='acc')
clf.fit(train_data)
predictions = clf.predict(test_data)
