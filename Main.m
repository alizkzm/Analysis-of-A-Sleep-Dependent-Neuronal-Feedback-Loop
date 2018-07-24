[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7022J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7022JM-Hypnogram_annotations.txt') ;

%% Part 3
%% Applying PCA and plotting the variance graph
[coeff,score,latent] = pca(X) ; % PCA on X matrix
Percentage = zeros (1,10) ;
Totall_Variance = sum (latent) ;
for i=1:10
   Percentage(i) = (sum(latent(1:i))/Totall_Variance) * 100 ; % Showing what percentage of the space being described by the eigenvectors
end
n = 1:10 ;
figure
plot (n , Percentage)
title ('Data Varinace')
xlabel ('N')
ylabel ('Percentage')
%% 

X1 = score (:,1) ;
X2 = score (:,2) ;
X3 = score (:,3) ;

% finding state indices

zero_state_ind = find (state == 0) ;
one_state_ind = find (state == 1) ;
two_state_ind = find (state == 2) ;
three_state_ind = find (state == 3) ;
four_state_ind = find (state == 4) ;
five_state_ind = find (state == 6) ;

% coefficionts of state 0

awake1 = X1(zero_state_ind) ;
awake2 = X2(zero_state_ind) ;
awake3 = X3(zero_state_ind) ;

% coefficionts of state 1

FirstState1 = X1(one_state_ind) ;
FirstState2 = X2(one_state_ind) ;
FirstState3 = X3(one_state_ind) ;

% coefficionts of state 2

SecondState1 = X1(two_state_ind) ;
SecondState2 = X2(two_state_ind) ;
SecondState3 = X3(two_state_ind) ;

% coefficionts of state 3

thirdState1 = X1(three_state_ind) ;
thirdState2 = X2(three_state_ind) ;
thirdState3 = X3(three_state_ind) ;

% coefficionts of state 4

fourthState1 = X1(four_state_ind) ;
fourthState2 = X2(four_state_ind) ;
fourthState3 = X3(four_state_ind) ;

% coefficionts of state 5

REM1 = X1(five_state_ind) ;
REM2 = X2(five_state_ind) ;
REM3 = X3(five_state_ind) ;

% Plotting the points

figure
plot3 (awake1, awake2 , awake3 , '.' , 'color' , 'b')
hold on 
plot3 (FirstState1 , FirstState2 , FirstState3 , '.' , 'color' , 'r')
hold on
plot3 (SecondState1 , SecondState2 , SecondState3 , '.' , 'color' , 'y')
hold on
plot3 (thirdState1 , thirdState2 , thirdState3 , '.' , 'color' , 'm')
hold on
plot3 (fourthState1, fourthState2 , fourthState3 , '.' , 'color' , 'g')
hold on
plot3 (REM1 , REM2 , REM3 , '.' , 'color' , 'c')

legend ('1','2','3','4')

xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
%% Plotting the points seperatly

figure
subplot(3,2,1)
plot3 (awake1, awake2 , awake3 , '.' , 'color' , 'b')
title ('awake')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
subplot(3,2,2)
plot3 (FirstState1 , FirstState2 , FirstState3 , '.' , 'color' , 'r')
title ('state 1')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
subplot(3,2,3)
plot3 (SecondState1 , SecondState2 , SecondState3 , '.' , 'color' , 'y')
title ('State 2')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
subplot(3,2,4)
plot3 (thirdState1 , thirdState2 , thirdState3 , '.' , 'color' , 'm')
title ('State 3')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
subplot(3,2,5)
plot3 (fourthState1, fourthState2 , fourthState3 , '.' , 'color' , 'g')
title ('State 4')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
subplot(3,2,6)
plot3 (REM1 , REM2 , REM3 , '.' , 'color' , 'c')
title ('REM')
xlabel ('PC1')
ylabel ('PC2')
zlabel ('PC3')
%% Part 4
%% Fitting a linear model on data

x = X ;
Wake_REM_index = find(state==0|state==6) ; % seperating the REM and awake data
x(Wake_REM_index,:) = [] ; % deletting the REM and awake data
y = (state)' ;
y(Wake_REM_index,:) = [] ;
LinMod = fitlm (x,y) % Regression Model

%% Plotting histogram on states 1_4
figure
histogram(LinMod.Fitted(find(y==1)),'Normalization','pdf','BinWidth',0.05)
hold on
histogram(LinMod.Fitted(find(y==2)),'Normalization','pdf','BinWidth',0.05) 
hold on
histogram(LinMod.Fitted(find(y==3)),'Normalization','pdf','BinWidth',0.05) 
hold on
histogram(LinMod.Fitted(find(y==4)),'Normalization','pdf','BinWidth',0.05)
legend ('state1','state2','state3', 'state4')
%% Plotting histogram on REM and awake states
figure
histogram(LinMod.predict(X(find(state==0),:)),'Normalization','pdf','BinWidth',5)
hold on
histogram(LinMod.predict(X(find(state==6),:)),'Normalization','pdf','BinWidth',5) 
legend ('Awake' , 'REM')
%% Part 5 _ SVM on raw data
[t2 ,state2 ,X2] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7022J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7022JM-Hypnogram_annotations.txt') ;
Xc=[x y]; %training data
[trained,v1]=trainClassifier(Xc); 
testclassifier=trained.predictFcn(x); % prediction of classifier
x2 = X2 ;
Wake_REM_index = find(state==0|state==6) ;
x2(Wake_REM_index,:) = [] ;
y2 = (state2)' ;
y2(Wake_REM_index,:) = [] ;
a=trained.predictFcn(x2);
ca=confusionmat(y2,a)
c=confusionmat(y,testclassifier)
%% SVM on data after applying PCA
[trained2,v2]=trainClassifier2(Xc);
testclassifier2=trained2.predictFcn(x) ;
c2=confusionmat(y,testclassifier2)
%% age and states
Num_state1 = zeros (5,1) ;
Num_state2 = zeros (5,1) ;
Num_state3 = zeros (5,1) ;
Num_state4 = zeros (5,1) ;
Num_stateREM = zeros (5,1) ;
[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7011J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7011JP-Hypnogram_annotations.txt') ;
Num_state1(1) = (length(find(state==1))*100)/length (state) ;
Num_state2(1) = (length(find(state==2))*100)/length (state) ;
Num_state3(1) = (length(find(state==3))*100)/length (state) ;
Num_state4(1) = (length(find(state==4))*100)/length (state) ;
Num_stateREM(1) = (length(find(state==6))*100)/length (state) ;
[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7022J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7022JM-Hypnogram_annotations.txt') ;
Num_state1(2) = (length(find(state==1))*100)/length (state) ;
Num_state2(2) = (length(find(state==2))*100)/length (state) ;
Num_state3(2) = (length(find(state==3))*100)/length (state) ;
Num_state4(2) = (length(find(state==4))*100)/length (state) ;
Num_stateREM(2) = (length(find(state==6))*100)/length (state) ;
[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7041J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7041JO-Hypnogram_annotations.txt') ;
Num_state1(3)= (length(find(state==1))*100)/length (state) ;
Num_state2(3) = (length(find(state==2))*100)/length (state) ;
Num_state3(3) = (length(find(state==3))*100)/length (state) ;
Num_state4(3) = (length(find(state==4))*100)/length (state) ;
Num_stateREM(3) =(length(find(state==6))*100)/length (state) ;
[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7052J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7052JA-Hypnogram_annotations.txt') ;
Num_state1(4) = (length(find(state==1))*100)/length (state) ;
Num_state2(4) = (length(find(state==2))*100)/length (state) ;
Num_state3(4) = (length(find(state==3))*100)/length (state) ;
Num_state4(4) = (length(find(state==4))*100)/length (state) ;
Num_stateREM(4) = (length(find(state==6))*100)/length (state) ;
[t state X] = FeatureExtraction ('C:\Users\Alireza\Desktop\Project2\Data\ST7061J0-PSG.edf','C:\Users\Alireza\Desktop\Project2\Data\ST7061JR-Hypnogram_annotations.txt') ;
Num_state1(5) = (length(find(state==1))*100)/length (state) ;
Num_state2(5) = (length(find(state==2))*100)/length (state) ;
Num_state3(5) = (length(find(state==3))*100)/length (state) ;
Num_state4(5) = (length(find(state==4))*100)/length (state) ;
Num_stateREM(5) = (length(find(state==5))*100)/length (state) ;
age = [60 35 18 32 35] ;
figure
stem (age , Num_state1)
xlabel ('age')
ylabel ('State 1 percentage')
title ('Percentage of state1 sleep versus age')
figure
stem (age , Num_state2)
xlabel ('age')
ylabel ('State 2 percentage')
title ('Percentage of state2 sleep versus age')
figure
stem (age , Num_state3)
xlabel ('age')
ylabel ('State 3 percentage')
title ('Percentage of state3 sleep versus age')
figure
stem (age , Num_state4)
xlabel ('age')
ylabel ('State 4 percentage')
title ('Percentage of state4 sleep versus age')
figure
stem (age , Num_stateREM)
xlabel ('age')
ylabel ('REM percentage')
title ('Percentage of REM sleep versus age')