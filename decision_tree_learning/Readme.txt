DECISION TREE LEARNING
~~
Mining of important properties of the model depends on some set of rules that
the model follows.  Here, we need to use an algorithm which is based on con-ditional 
probabilities and generates rules which are relevant to the model.  For this purpose, 
we choose the Decision Tree algorithm for mining the important properties of the model.  
Decision trees generate rules.  A rule is a conditional statement that can easily be 
understood by humans and easily used within a database to identify a set of records of 
satisfying sample data points.  We aim to  find  these  rules  for  the  training  dataset
that  we  train  our SCAD baseline neural network model on and analyse as to how many rules 
actually hold true on the testing dataset. These rules will constitute as the important mined 
properties of  the  model  and  hence  later  on  can  be  used  as  queries  through  Marabou  
to assess the robustness of the model on the test set.
~~
DecisionTreeLearning_25Features.ipynb consists of code for training, displaying the decision 
tree and the corresponding rules for all 25 features in the training dataset. This notebook also
consists of code to convert the scaled rules back to the raw/original values. It also has a small 
code to show the most important features in the training dataset using random forests.
~~
DecisionTreeLearning_24Features.ipynb consists of the code for training, displaying the decision tree
and the rules for 24 features, excluding the ManualWheel feature from the original training dataset.
~~
Rules_25Features.txt and Rules_24Features.txt consists of the rules extracted from each experiment 
mentioned above.
~~
