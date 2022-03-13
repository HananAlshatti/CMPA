Isr=0.01e-12; 
Ibb =0.1E-12  
Vbb =1.3; 
Gp =0.1; 

 
VectorforVoltage= linspace(-1.95,0.7,200);

I=Isr .*(exp((1.2/0.025).*VectorforVoltage)-1) + Gp.*VectorforVoltage - Ibb.*(exp((-1.2/0.025).*(VectorforVoltage +Vbb))-1);

Variance = (rand(1,200)*.20);

Noise= I.*Variance;
I_noise= I + Noise; 

figure
subplot(3,2,1)
plot(VectorforVoltage,I_noise)
title ('V vs I')
subplot(3,2,2)
semilogy(VectorforVoltage, abs(I_noise))
title('semiwith polyfit')
 
fourth=polyfit(VectorforVoltage, I_noise,4) 
I_4=polyval(fourth, VectorforVoltage);

eight=polyfit(VectorforVoltage, I_noise,8)
I_8=polyval(eight,VectorforVoltage);
subplot(3,2,1);
hold on 
plot(VectorforVoltage,I_4)
plot(VectorforVoltage,I_8); 

subplot(3,2,2)
hold on 
semilogy(VectorforVoltage,abs(I_4))
semilogy(VectorforVoltage, abs(I_8))

fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1 = fit(transpose(VectorforVoltage),transpose(I),fo1);
If1 = ff1(VectorforVoltage);
subplot(3, 2, 3)
plot(VectorforVoltage, If1);
title('Non-linear fit')
hold on
subplot(3, 2, 4)
semilogy(VectorforVoltage, abs(If1));
title('Semilogy Non-linear fit')
hold on

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(transpose(VectorforVoltage),transpose(I),fo2);
If2 = ff2(VectorforVoltage);
subplot(3, 2, 3)
plot(VectorforVoltage, If2);
hold on
subplot(3, 2, 4)
semilogy(VectorforVoltage, abs(If2));
hold on

fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(transpose(VectorforVoltage),transpose(I),fo3);
If3 = ff3(VectorforVoltage);
subplot(3, 2, 3)
plot(VectorforVoltage, If3);
hold on
subplot(3, 2, 4)
semilogy(VectorforVoltage, abs(If3));
hold on

inputs = VectorforVoltage;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

figure (2)
plot(VectorforVoltage, Inn);
title('NN fit')

figure(3)
semilogy(VectorforVoltage, abs(Inn));
title('fit')