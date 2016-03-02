clear;
% clc;
% close all;

%% Getting the Input and Output

im =textread('Data.txt','%s');
inp = [];
out = [];
% lm = 1;
height = size(im,1);
for i = 33:33:height
    if (im{i} == '0')
        inp = [inp;im((i-32):(i-1))];
        out = [out;[1 0]];  
    end
    if (im{i} == '7')
        inp = [inp;im((i-32):(i-1))];
        out = [out;[0 1]];
    end
end

heigh = size(inp,1);
inpu = zeros(heigh,32);
for i = 1:heigh
    vary = inp{i};
    for j = 1:32
        inpu(i,j) = str2double(vary(j));
    end
end

input = [];

for i = 1:32:(heigh-31)
    imag = inpu((i:i+31),:);
    temp1 = imresize(imag,1/4,'bilinear');
    input = [input;temp1(:)'];
end

input_old = input;
input = uint8(input);
% patterns = patterns(1:80,:);
target_out = out;
% desired_out = desired_out(1:80,:);

%% Initialization of variables

sse = 1000;    % sse is the Sum Squared Error   
eta = 1;                       
alpha = 0.8; 
sse_rec = [];  
input = [input ones(size(input,1),1) ];       
number_of_inputnodes = size(input,2);     
number_of_hiddennodes = 5;                    
number_of_outputnodes = size(target_out,2);    

%% Initialization of Weights

%w1 has the weights from i-th input node to the j-th hidden node
axp1 = (-1)./(sqrt(64));
bxp1 = (1)./(sqrt(64));
w1 = 1 .* ((bxp1-axp1).*rand(number_of_inputnodes,number_of_hiddennodes - 1) + axp1);

%w2 has Weight from j-th hidden node to the k-th output node
axp2 = (-1)./(sqrt(number_of_hiddennodes));
bxp2 = (1)./(sqrt(number_of_hiddennodes));
w2 = 1 .* ((bxp2-axp2).*rand(number_of_hiddennodes,number_of_outputnodes) + axp2);
       
last_delta_w1 = zeros(size(w1));             % last change in w1
last_delta_w2 = zeros(size(w2));             % last change in w2
epoch = 0;                              

%% Computation  

while sse > 0.1                        
        input_to_hidden = (double(input)) * w1;  
        hidden_act = 1./(1+exp( - input_to_hidden)); 
        hidden_with_bias = [ hidden_act ones(size(hidden_act,1),1) ];    
        hidden_to_output = hidden_with_bias * w2; 
        output_act = 1./(1+exp( - hidden_to_output)); 
        output_error = target_out - output_act;   
        sse = trace(output_error' * output_error); 
        sse_rec = [sse_rec sse]; 
        
        delta_hidden_to_output = output_error .* output_act .* (1-output_act);                                        
        delta_input_to_hidden = delta_hidden_to_output*w2' .* hidden_with_bias .* (1-hidden_with_bias);
        delta_input_to_hidden(:,size(delta_input_to_hidden,2)) = [];  
               
        % Backpropagation
        dw1 = eta * (double(input))' * delta_input_to_hidden + alpha .* last_delta_w1;   
        dw2 = eta * hidden_with_bias' * delta_hidden_to_output + alpha .* last_delta_w2;
        
        w1 = w1 + dw1; 
        w2 = w2 + dw2;    
        
        last_delta_w1 = dw1; 
        last_delta_w2 = dw2;         
        
        epoch = epoch + 1;
        if (epoch == 20000)
            break;
        end
end     

w1
w2
output_act;
acdeg = round(output_act);
ans = sum(sum(acdeg - target_out));

