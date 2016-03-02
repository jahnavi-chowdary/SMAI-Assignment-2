% clear;
% clc;
% close all;

% initial_weight = rand(1,3);
initial_weight = [1 1 1];

%% Single Sample Perceptron Without Margin

w = initial_weight;
initial_weight
w = w';

% Load the Data Points
class = [];

class1 = [1 7 1;6 3 1;7 8 1;8 9 1;4 5 1;7 5 1];
class2 = [3 1 1;4 3 1;2 4 1;7 1 1;1 3 1;4 2 1];

class = [class; class1];
class = [class; -class2];
n = size(class,1);

%Randomize
randorder = randperm(n);
rand_class = class(randorder, :);

l = 0;
flag = 1;
stop = 0;

while(flag == 1 && l<10000)
    l = l +1;
    flag = 0;

    for k = 1:12
        value = rand_class(k,:) * w;
        if value < stop
            flag = 1;
            w = w + rand_class(k,:)';
        end
    end
end

disp('For Single sample perceptron without margin');
l
w'

%Plotting
figure(1);
plot(class1(:,1),class1(:,2),'or');
hold on;
plot(class2(:,1),class2(:,2),'+b');
hold on;
x = [1 2 3 4 5 6 7 8 9 10];
y = -(w(1)*x + w(3))/w(2);
plot(x,y,'k');
hold on


%% Single Sample Perceptron With Margin

w = initial_weight;
w = w';

% Load the Data Points
class = [];

class1 = [1 7 1;6 3 1;7 8 1;8 9 1;4 5 1;7 5 1];
class2 = [3 1 1;4 3 1;2 4 1;7 1 1;1 3 1;4 2 1];

class = [class; class1];
class = [class; -class2];
n = size(class,1);

%Randomize
randorder = randperm(n);
rand_class = class(randorder, :);

l = 0;
flag = 1;
stop = 0.01;

while(flag == 1 && l<10000)
    l = l +1;
    flag = 0;

    for k = 1:12
        value = rand_class(k,:) * w;
        if value < stop
            flag = 1;
            w = w + rand_class(k,:)';
        end
    end
end

disp('For Single sample perceptron with margin');
l
w'

%Plotting
y = -(w(1)*x + w(3))/w(2);
plot(x,y,'c');
hold on

%% Relaxation Procedure

w = initial_weight;
w = w';

% Load the Data Points
class = [];

class1 = [1 7 1;6 3 1;7 8 1;8 9 1;4 5 1;7 5 1];
class2 = [3 1 1;4 3 1;2 4 1;7 1 1;1 3 1;4 2 1];

class = [class; class1];
class = [class; -class2];
n = size(class,1);

%Randomize
randorder = randperm(n);
rand_class = class(randorder, :);

wprev = zeros(1,3)';
stop = 1;
eta = 1;
thres = 0;
l=0;
flag = 1;

while (flag && pdist([w';wprev'])>thres)
   l = l+1;
   flag = 0;
   for k=1:12
      yk = rand_class(k,:);
      if yk * w <= stop
         value = yk*(stop-(yk * w))/power(pdist([zeros(1,3);yk]),2);
         wprev = w;
         w = w + eta * value';
         flag = 1;
      end
   end
   if flag==1
     eta=eta*0.9;
   end 
end

disp('For Relaxation Procedure');
l
w'

%Plotting
y = -(w(1)*x + w(3))/w(2);
plot(x,y,'m');
hold on;


%% Widrow-Hoff Algorithm


w = initial_weight;
w = w';

% Load the Data Points
class = [];

class1 = [1 7 1;6 3 1;7 8 1;8 9 1;4 5 1;7 5 1];
class2 = [3 1 1;4 3 1;2 4 1;7 1 1;1 3 1;4 2 1];

class = [class; class1];
class = [class; -class2];
n = size(class,1);

%Randomize
randorder = randperm(n);
rand_class = class(randorder, :);


wprev=zeros(1,3)';
thres=0.0001;
eta = 0.7;
flag = 1;
stop = 0.1 * ones(1,12);
l = 0;

while (flag && pdist([w';wprev'])>thres)
   l = l+1;
   flag = 0;
   for k=1:12
      yk = rand_class(k,:);
      if yk * w<=stop(k)
         value = yk*(stop(k)-(yk*w));
         wprev = w;
         w = w + eta*value';
         flag = 1;
      end
   end
   if flag==1
     eta=eta*0.9;
   end 
   
end

disp('For Widrow Hoff Algorithm');
l
w'

%Plotting
y = -(w(1)*x + w(3))/w(2);
plot(x,y,'y');
hold off

%%

legend('class1','class2','ssp','sspm','relax','widrow','Location','northwest');
xlabel('X');
ylabel('Y');
title('classifier');
hold off


