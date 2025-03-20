%% main_code.m

%% loading datas
inv_dt=[ %% 인버터 상세 데이터
    readmatrix('발전왕 비즈_집인태양광발전소_장비별데이터_2024-08-11.xlsx', 'Range', 'C25:H92'); 
    readmatrix('발전왕 비즈_집인태양광발전소_장비별데이터_2024-09-22.xlsx', 'Range', 'C25:H92');
    readmatrix('발전왕 비즈_집인태양광발전소_장비별데이터_2024-10-22.xlsx', 'Range', 'C25:H92'); %% C D E : 입력전압 입력전류 입력전력
    readmatrix('발전왕 비즈_집인태양광발전소_장비별데이터_2024-11-22.xlsx', 'Range', 'C25:H92')]; %% F G H : 출력전압 출력전류 출력전력

P_table=[ %% 발전량 데이터
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2023-12-22.xlsx', 'Range', 'B11:D26'); % 05시~20시 데이터
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-01-22.xlsx', 'Range', 'B11:D26'); % 발전량, 발전시간, 인버터1 발전량
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-02-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-03-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-04-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-05-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-06-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-07-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-08-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-09-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-10-22.xlsx', 'Range', 'B11:D26'); 
    readmatrix('발전왕 비즈_집인태양광발전소_일발전량_2024-11-22.xlsx', 'Range', 'B11:D26')     ];

weather=[ %% 시간 당 기온(℃), 일사량(MJ/m^2)
    readmatrix('1222영광.xlsx', 'Range', 'D3:E18'); % 05시~20시 데이터
    readmatrix('0122영광.xlsx', 'Range', 'D3:E18');
    readmatrix('0222영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0322영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0422영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0522영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0622영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0722영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0822영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('0922영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('1022영광.xlsx', 'Range', 'D3:E18'); 
    readmatrix('1122영광.xlsx', 'Range', 'D3:E18')      ];

weather(isnan(weather))=0; % 측정되지 않은 시점은 0으로 만든다.

day_simul='0322'; % 시뮬레이션 할 날짜
clock1st=5; % 시작 시각
smpl_wthr=readmatrix([day_simul,'영광.xlsx'], 'Range', 'D3:E18');  % 시뮬레이션 기상 데이터
smpl_wthr(:,2)=smpl_wthr(:,2)*10000/36;
smpl_wthr(:,2)=[ smpl_wthr( 2 : size( smpl_wthr(:,2) ,1) , 2 ); 0 ];
smpl_wthr(isnan(smpl_wthr))=0;


%% fitting efficiency curve

cstexp= fittype('a*exp(b*x) + c*x+d', 'independent', 'x', 'coefficients', {'a', 'b', 'c','d'}); %%
cstrat= fittype('a/x+b', 'independent', 'x', 'coefficients', {'a', 'b'});
%fittedmodel=fit(Rload,Eff,cstexp) %%근사 효율곡선
%fitcv=fit(Rload,Eff,'exp2'); % 이차 지수함수 형식 (지수함수+지수함수)
%fitcv2=fit(Rload,Eff,cstexp); % 사용자 지정 형식 (지수함수+일차함수+상수)
%fitcv3=fit(Rload(Rload ~= 0),Eff(Rload ~= 0),cstrat); % 사용자 지정 형식 (유리함수)
%그러나 세 형식 다 올바르게 커브피팅이 되지 않아 수동으로 조절해 efcv2로 지정했다..


%% setting datas

%% 초기 값 세팅

Vmaxx=65;
n=150; % 연산 step


prob_lv1=  10  / 100; % 85% 패널 존재 확률 -> 바이패스 안함
prob_lv2=  2  / 100; % 50% 패널 존재 확률
prob_lv3=  1  / 100; % 20% 패널 존재 확률
srsn=18; % 직렬 개수
prln=17; % 병렬 개수
boxn=8; % 센트럴 인버터 접속반 총 개수
strn=prln*boxn; % 총 스트링 개수
invmax=515e3; % 센트럴 인버터 정격 전압
invn=2; % 센트럴 인버터 개수

cst_csmP=3.7e-4; % 가동전력 = 정격전력 * cst_csmP (비례계수)


timeea=length(smpl_wthr(:,1)); % 시간 개수 (테스트 데이터의 길이)

V = linspace(0, Vmaxx, n); % 연산 기준 전압
maskG=ones(strn,srsn,timeea); % 각 패널에 할당되는 일사량
maskT=ones(strn,srsn,timeea); % 각 패널에 할당되는 온도


% 랜덤 마스크 (17,18,timeea) 0~1 선형 확률 부여.
% ex) 0~0.067 계수 가진 패널에 50%만 주기
% ex) 0.067~0.1 계수 가진 패널에 20%만 주기

position_Prtshd=ones(strn/4,srsn*4,timeea); % 부분음영 계수 (position 관련 값 비범용적 정의. 추후 보완 필요)
position_isshd=zeros(strn/4,srsn*4,timeea);  % 부분음영 여부
position_rndmask=rand(strn,srsn,timeea); % 랜덤 계수 행렬


condition1=position_rndmask >= 0 & position_rndmask < prob_lv1; % Lv1 음영 조건
condition2=position_rndmask >= prob_lv1 & position_rndmask < prob_lv1+prob_lv2; % Lv2 음영 조건
condition3=position_rndmask >= prob_lv1+prob_lv2 & position_rndmask < prob_lv1+prob_lv2+prob_lv3; % Lv3 음영 조건

Lv1shd = 0.85; % Lv1 음영 강도
Lv2shd = 0.5;
Lv3shd = 0.2;

position_Prtshd(condition1) = Lv1shd; % Lv1 음영 정도
position_Prtshd(condition2) = Lv2shd; % Lv2 음영 정도
position_Prtshd(condition3) = Lv3shd; % Lv3 음영 정도

position_isshd(condition1)=1; % 음영 여부 마스크 행렬 갱신
position_isshd(condition2)=2; % 음영 여부 마스크 행렬 갱신
position_isshd(condition3)=3;

for tt=1:timeea % 기상 데이터 제작
    maskG(:,:,tt)=ones(strn,srsn)*smpl_wthr(tt,2);
    maskT(:,:,tt)=ones(strn,srsn)*smpl_wthr(tt,1);
end

radius =20;

istherecloud = 0; % 구름 유무 결정
cst_clound=ones(1,timeea); % 구름 발생으로 인해 다이나믹 인버터의 에너지 손실 계수
cloud_time=[9 11 12 16 17]; %5시~20시 사이 값들 삽입

if istherecloud==1
    for tt = 1:timeea % 시간에 따른 구름 음영 이동
        if any(cloud_time==tt+clock1st-1)

            cst_clound(tt)=1 - 9*pi*(radius)^2/(16*strn*srsn)*size(cloud_time,2)/timeea;

            center_x = srsn*4*tt/timeea; % 구름 중심 (시간에 따라 이동)
            center_y = strn*tt/(4*timeea); 
        
            for st = 1:strn/4 % 범용적이지 않음. 추후 보완
                for sr = 1:srsn*4
                    distance = sqrt((sr - center_x)^2 + (st - center_y)^2); % 연산 위치 거리 계산
                    
                    if distance <= radius % 음영 범위 내
                        if distance <= radius/2 % 가장 가운데
                            position_Prtshd(st, sr, tt) = Lv3shd; 
                            position_isshd(st, sr, tt) = 4;
        
                        elseif distance <= 3*radius/4 && position_isshd(st,sr,tt)<=1
                            position_Prtshd(st, sr, tt) = Lv2shd; % 중간 부분
                            position_isshd(st, sr, tt) = 2;
        
                        elseif position_isshd(st,sr,tt)==0
                            position_Prtshd(st, sr, tt) = Lv1shd; % 가장자리
                            position_isshd(st, sr, tt) = 1;
        
                        end
                    end
                end
            end
        end
    end
end

Prtshd=[position_Prtshd(:,1:18,:);position_Prtshd(:,19:36,:);position_Prtshd(:,37:54,:);position_Prtshd(:,55:72,:)];
isshd=[position_isshd(:,1:18,:);position_isshd(:,19:36,:);position_isshd(:,37:54,:);position_isshd(:,55:72,:)];


%for tt=1:timeea % 기상 데이터 제작
%    maskG(:,:,tt)=ones(strn,srsn)*(8-abs(tt-8))*120;
%    maskT(:,:,tt)=ones(strn,srsn)*(30-2.5*abs(tt-9));
%end


Pin=inv_dt(:,3); %% 인버터 직류입력전력
Pout=inv_dt(:,6); %% 인버터 교류출력전력

Eff=Pout./Pin*100; %%변환효율=교류출력/직류입력
    Eff(isnan(Eff)) = 0; %%입력이 0일 때 NaN 발생 -> 0 초기화

Rload=Pin./invmax*100000; %%부하율 (입력 직류전력/정격전력)

table=[Pin,Pout,Eff,Rload];
table=sortrows(table,4); %%부하율 기준 정렬


colors = 1-jet(timeea); % 시간별로 그래프 색을 구분하기 위해 가장 적합하다고 판단된 무지개 색의 반전을 사용


%% 부하율 관련 데이터 plot

figure;

%%subplot(2,2,1)
area(Rload, 'FaceColor', '#C1D8C3'); % 꺾은선 그래프 아래를 색칠
hold on;
plot(Rload,'Color','#6A9C89','LineWidth',1)
title('인버터 부하율 (입력 직류전력/정격전력)')
axis([0,272,0,100])
grid on 
xlabel('월/일 (2024년)'); ylabel('부하율 (%)')
xticks([36 100 167 237]); % x축 값 없애기
xticklabels({'8/22', '9/22', '10/22', '11/22'}); % 해당 위치에 라벨 지정
xline([68.5,136.5,204.5], '--k', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'right'); 
x=0:0.5:100; % efcv2 함수 연속적으로 나타내기 위한 정의역

%%

figure;
%subplot(2,2,2)
plot(table(:,4),table(:,3),'o','Color','#6A9C89') % 데이터 산점도 
hold on
%plot(x,fitcv(x),'LineWidth',1.5,'Color', '#FF4545'); hold on;
%plot(x,fitcv2(x),'LineWidth',1.5,'Color', '#FF9C73'); hold on;
%plot(x,fitcv3(x),'LineWidth',1.5,'Color', '#FBD288'); hold on;

%%plot(x,efcv2(x),'LineWidth',1,'Color', '#CD5C08'); % 효율곡선
axis([0,100,0,130]) %유의미한 부분만 보여준다.
title('산점도에 대한 curve fitted 변환효율곡선')
grid on
xlabel('부하율 (%)'); ylabel('변환효율 (%)')
legend('실데이터 산점도','curve fitted (이차지수 형식)','curve fitted (지수+직선 형식)','curve fitted (유리+상수 형식)')

%%
figure;
%subplot(2,2,2)
plot(table(:,4),table(:,3),'o','Color','#6A9C89') % 데이터 산점도 
hold on
%plot(x,efcv(x),'LineWidth',1.5,'Color', '#FF4545'); hold on;
plot(x,efcv2(x),'LineWidth',1.5,'Color', '#FF9C73'); hold on;
    
%%plot(x,efcv2(x),'LineWidth',1,'Color', '#CD5C08'); % 효율곡선
axis([0,100,0,110]) %유의미한 부분만 보여준다.
title('산점도에 대한 curve fitted 변환효율곡선')
grid on
xlabel('부하율 (%)'); ylabel('변환효율 (%)')
legend('실데이터 산점도','curve fitted (지수+직선 형식)','curve fitted (유리+상수 형식)')

%%

figure;
%subplot(2,2,3)
yyaxis left
histogram(Rload(Rload~=0),'BinWidth',4,'Normalization', 'probability','FaceColor', '#C1D8C3','EdgeColor', '#6A9C89','LineWidth',0.8); %0이 아닌 값들에 대한 부하율 히스토그램
hold on
yticks = get(gca, 'YTick'); % 현재 y축 눈금 값을 가져옴
set(gca, 'YTick', yticks, 'YTickLabel', yticks * 100); % y축 눈금 값을 % 단위로 변환
ylabel('발전시간 대비 차지 비율 (%)')

yyaxis right
plot(x,efcv2(x),x,x./x+99,'LineWidth', 1,'Color', '#CD5C08') % 효율곡선 겹치기
ylim([0,115])
title('부하율 분포 히스토그램 (부하율 0%인 경우는 제외)')
xlabel('부하율 (%) (4%p 간격)'); ylabel('변환효율 (%)')
grid on;
legend('부하율 분포','변환효율')


%% PredictP.m
%% loading data

P_gen=P_table(:,1); % 발전량 슬라이싱
inv_load=P_table(:,3); % 인버터 발전량 슬라이싱


%%

Tv=weather(:,1); %%온도 벡터
Gv=weather(:,2)*10000/36; % 일사량 벡터 (W/m^2)
Gv=[Gv(2:size(Gv,1));0]; % 기존데이터가 (J) 단위로, '누적된 값'을 보여줬으므로 한칸 미뤄 (W)에 맞게 전처리




cst_b=-2;%%(%/K) <- 최적의 온도계수
cst_b2=-0.35;%%(%/K) <- 시행착오 용도
cst_b3=+0.9;%%(%/K)
P_mr=415;%%(W) 기준 정격전압
G_r=1000;%%(W/m2) 기준 일사량
T_r=25;%%(℃) 기준 온도

P_m = P_mr * Gv / G_r .* ( 1 + cst_b /100 * ( Tv - T_r )); % 최대 발전량 예측 벡터
tt=1:size(Gv,1);

figure;
subplot(2,1,1)

ax1 = axes; % 첫 번째 y축

area(ax1,tt,P_gen/2,'FaceColor','#F6D6D6', 'FaceAlpha', 0.8,'HandleVisibility','off'); % 발전량 그래프 색칠
hold on;
area(ax1,tt,P_m,'FaceColor', '#7BD3EA', 'FaceAlpha', 0.3,'HandleVisibility','off') % 예측 그래프 색칠
hold on;
plot(ax1,tt,P_m,'LineWidth', 1,'Color','#37C6FF','HandleVisibility','off' )
hold on;
plot(tt,P_gen/2,'LineWidth', 1,'Color','#F68282','HandleVisibility','off');
axis([0,195,0.1,415])

ylabel(ax1, '발전량 (kW)')
xticks([]); % x축 값 없애기

ax2 = axes('Position', get(ax1, 'Position')); % 두 번째 축, 동일한 위치에 투명하게 설정
ax2.YColor='#357849';
ax2.YAxisLocation = 'right'; % 오른쪽 y축 설정
hold on;
plot(tt,Tv,'LineWidth', 1,'Color','#A1EEBD','HandleVisibility','off') % 온도 그래프
ylabel(ax2,'온도 (℃')
set(ax2, 'XAxisLocation', 'top', 'Color', 'none');
xticks([]); % x축 값 없애기
axis([0,195,-20,40])

ax3 = axes('Position', get(ax1, 'Position') + [0, 0,0, 0], 'Color', 'none'); % 약간 오른쪽으로 이동하고 너비 조정
ax3.YAxisLocation = 'right'; % 오른쪽 y축 설정f
ax3.YColor='#CFA410';
hold on;
plot(tt,Gv,'LineWidth', 1,'Color','#FFB200','HandleVisibility','off') % 일사량 그래프

hold on;

plot(tt,tt./tt-10,'LineWidth', 1,'Color','#37C6FF','DisplayName','발전량'); hold on;
plot(tt,tt./tt-10,'LineWidth', 1,'Color','#F68282','DisplayName','예측 발전량'); hold on;
plot(tt,tt./tt-10,'LineWidth', 1,'Color','#A1EEBD','DisplayName','온도'); hold on;
plot(tt,tt./tt-10,'LineWidth', 1,'Color','#FFB200','DisplayName','일사량'); hold on;
legend("show")

ylabel(ax3,'일사량(W/m^2)')
set(ax3, 'XTick', [], 'Color', 'none');
axis([0,195,0,1000]);
grid on
xlabel('2024년        12/22             01/22            02/22              03/22              04/22              05/22             06/22             07/22             08/22             09/22             10/22             11/22    발전량       ')
title('일사량, 기온을 기반으로한 발전량 예측 모델')
xticks([]); % x축 값 없애기
legend('show')

%% PlottingIVcurve.m

Gminn=200; % IVcurve의 최대/최소 일사/온도 값
Gmaxx=1000;
Tminn=-10;
Tmaxx=30;
G=linspace(Gminn,Gmaxx,n); %% T, G, V의 개수가 같아야한다.
T=linspace(Tminn,Tmaxx,n);
%%
[Vm,Gm]=meshgrid(V,G);
I=solarIV(Gm,Vm,28);

figure;
subplot(2,2,1)
mesh(V,G,I)
axis([0,Vmaxx,Gminn,Gmaxx,0,15])
xlabel('전압 (V)'), ylabel('일사량 (W/m^2)'), zlabel('전류 (A)')
title('일사량 변화에 따른 I-V 곡선 (28℃)')
grid on

%%
[Vm,Tm]=meshgrid(V,T);
I=solarIV(800,Vm,Tm);

subplot(2,2,2)
mesh(V,T,I)
axis([0,Vmaxx,Tminn,Tmaxx,0,15])
xlabel('전압 (V)'), ylabel('온도 (℃)'), zlabel('전류 (A)')
title('온도 변화에 따른 I-V 곡선 (800W/m^2)')
grid on
colorbar;


%%
subplot(2,2,3)
I=solarIV(Gm,Vm,28);

P = I .* V; % P = I * V
% P-V 곡선의 최대 지점을 계산하고 연결
P_max_values = max(P, [], 2); % 각 일사량(G)에서 최대 전력 값
[V_max_idx] = arrayfun(@(i) find(P(i,:) == P_max_values(i), 1), 1:size(P,1)); % 최대값의 인덱스 찾기
V_max_values = V(V_max_idx); % 최대 전력 지점에 대응되는 전압 값


mesh(V,G,I.*V);
axis([0,Vmaxx,Gminn,Gmaxx,0,600])
xlabel('전압 (V)'), ylabel('일사량 (W/m^2)'), zlabel('전력 (W)')
title('일사량 변화에 따른 P-V 곡선 (28℃)')
grid on
hold on;
plot3(V_max_values, G, P_max_values, 'LineWidth', 2, 'Color', '#F68282');



for k = 1:length(G)
    % 최대점에서 G-V 평면으로 수직선 그리기
    plot3([V_max_values(k), V_max_values(k)], [G(k), G(k)], [0, P_max_values(k)], 'Color','#F68282');
end 

%%
subplot(2,2,4)
I=solarIV(800,Vm,Tm); 

P = I .* V;
P_max_values = max(P, [], 2);
[V_max_idx] = arrayfun(@(i) find(P(i,:) == P_max_values(i), 1), 1:size(P,1));
V_max_values = V(V_max_idx); 

mesh(V,T,I.*V)
axis([0,Vmaxx,Tminn,Tmaxx,0,600])
xlabel('전압 (V)'), ylabel('온도 (℃)'), zlabel('전력 (W)')
title('온도 변화에 따른 P-V 곡선 (800W/m^2)')
grid on
hold on;
plot3(V_max_values, T, P_max_values, 'LineWidth', 2, 'Color', '#F68282');

colorbar;

for k = 1:length(T)
    % 최대점에서 G-V 평면으로 수직선 그리기
    plot3([V_max_values(k), V_max_values(k)], [T(k), T(k)], [0, P_max_values(k)], 'Color','#F68282');
end 



%% 기상 데이터 제작


pureG=maskG; % 음영 가해지기 전 순수한 일사량
maskG=maskG.*Prtshd; % 음영 가하기
avgG=squeeze(mean(mean(maskG,1),2)); % 전체 일사량 평균
avg_pureG = zeros(1, timeea); % 순수 일사량만의 평균 공간 사전제작

for tt = 1:timeea  
    % shade_condition = find(isshd(:,:,tt)>=2) % Lv2와 Lv3가 아닌 패널에 대해 (Lv1 포함)

    tmp = maskG(:, :, tt); % maskG에 조건 가할 시 복잡하므로 연산 오류를 피하기 위해 임시 저장
    avg_pureG(tt) = mean(tmp(isshd(:,:,tt)<=1)); % 조건에 해당하는 값들의 평균
end

%% 바둑판 형태로 일사량 시각화
figure;
cellea = ceil(sqrt(timeea)); % 한변 그래프 개수를 "시간 개수의 제곱근의 올림"으로 지정 (기하평균 사용)

max_val = 1050; % 모든 그래프에 같은 컬러바를 설정하기 위해 최댓값 찾기


position_G=[maskG(1:34,:,:),maskG(35:68,:,:),maskG(69:102,:,:),maskG(103:136, :, :)]; % 패널의 실제 배치

for tt = 1:timeea
    subplot(cellea, cellea, tt);
    imagesc(position_G(:,:,tt)); % 2x4 배열로 접속반 플롯
    clim([0, max_val]); % 모든 서브플롯에 동일한 컬러 축 설정
    colormap(hot); % 일사량에 어울리는 컬러맵 설정
    line([18.5 18.5], ylim, 'Color', 'k', 'LineWidth', 1.5); hold on;
    line([36.5 36.5], ylim, 'Color', 'k', 'LineWidth', 1.5); hold on; 
    line([54.5 54.5], ylim, 'Color', 'k', 'LineWidth', 1.5); hold on; 
    line(xlim, [17.5 17.5], 'Color', 'k', 'LineWidth', 1.5); hold on;
    title([num2str(tt+clock1st-1), '시 (최고 일사량 : ',num2str(int32(smpl_wthr(tt,2))),'W/m^2)']) % 그래프마다 다른 제목 (시각)
    if tt==1
        title({['24',day_simul,' 전체 패널(36x72) 일사량'], ...
            ['85%음영은 ',num2str(prob_lv1*100),'% 존재'] , ['50%음영은 ',num2str(prob_lv2*100),'% 존재'] , ['20%음영은 ',num2str(prob_lv3*100),'% 존재']});
    end
end
hold on;
    % 세로 구획선을 그리기 (36패널을 기준으로)


% 공통 컬러바 설정
colorbar('Position', [0.92, 0.1, 0.02, 0.8]); % figure 외부에 컬러바 추가




%% 전압-전류-전력 연산 ***

Current=ones(n,strn,srsn,timeea); % 패널 전류 (기준전압별, 스트링 별, 시간 별)
I_box=zeros(n,timeea,boxn); % 접속반 전류 (시간별, 접속반별, 기준전압별)
strV=zeros(strn,n,timeea); % 스트링 전압 (스트링별, 기준전류별, 시간별)
I_std=zeros(n,timeea);

for tt=1:timeea
    fprintf("시간 :%d,   ",tt) % 가장 시간이 오래 걸리는 부분에서 진행상황을 보기 위해 연산 수행중인 시각을 출력
    for bx=1:boxn % 접속반 단위로 진행

%% 17x18 I-V 곡선 제작
        for st=1+prln*(bx-1):prln*bx % 접속반에 포함된 스트링에 대해
            for sr=1:srsn % 특정 스트링을 이루는 패널들에 대해
               Current(:,st,sr,tt)=solarIV(maskG(st,sr,tt),V,maskT(st,sr,tt)); % 모든 개별 패널에 대해 I-V 곡선 제작
            end
        end


%% 직렬합
        I_std(:,tt)=linspace(0.001,max(Current(1,:,:,tt), [], 'all'),n); % 기준전류
        
        V_itp=zeros(srsn,n); % 18직렬에서 n개의 기준 전류에 의해 측정된 내삽 전압들 공간 사전 제작
        % 직렬과 병렬 연결 간에 I-V 곡선 합산 방식이 상이하다 (전류에 대해 전압 합계/전압에 대해 전류 합계)
        % ex 전압 합산을 하려면 동일한 전류 위치에서 시행해야 하는데, I-V 곡선이 비선형이기에 기준점이 다 상이하다.
        % 그러므로 실 데이터가 아닌, 데이터 사이의 추정값을 이용해 직렬합을 구현했다.
      
        
        for strn=1+prln*(bx-1):prln*bx % 접속반의 스트링에 대해
            for sr=1:srsn % 직렬을 이루는 패널들에 대해
                [unqItmp,unqidx]=unique(Current(:,strn,sr,tt)); % 전류 중복값 제거 (내삽 시 문제 방지)
                unqVtmp=V(unqidx); % 중복값 제거된 전류와 같은 위상으로 전압 pick
                if length(unqItmp) > 1 && any(unqItmp > 0) % 피연산 대상 길이가 2 이상이고 전부 0이 아닐 때 (전류가 0인 부분에서 내삽할 점이 1개만 존재하면 오류 발생)
                    V_itp(sr, :) = interp1(unqItmp, unqVtmp, I_std(:,tt), 'linear'); % 내삽값
                    V_itp(sr, isnan(V_itp(sr, :))) = 0; % NaN 제거
                else
                    V_itp(sr, :) = zeros(1, n); % 유효한 데이터가 없으면 0으로 설정
                end
                
                strV(strn,:,tt)=strV(strn,:,tt)+V_itp(sr,:); % 스트링 전압 : 직렬 전압 합산

            end
        end
    
        
        %% 병렬합
        I_itp = zeros(prln*boxn, n); % 17 스트링에 대한 n 개 기준 전압에서의 전류 (합산된 스트링 전압을 또다시 전압에 대해 합산하기 위해)
    
        for strn = 1+prln*(bx-1):prln*bx % 스트링에 대해
            [unqVtmp, unqidx] = unique(strV(strn, :,tt)); % 중복값 제거
            unqItmp = I_std(unqidx,tt);
    
           
            if length(unqVtmp) > 1 && any(unqVtmp > 0)
                I_itp(strn, :) = interp1(unqVtmp, unqItmp, V*srsn, 'linear',0); % 보간 후 NaN 값은 0으로 처리
            else
                I_itp(strn, :) = zeros(1, n); % 유효한 데이터가 없으면 0으로 설정
            end
    
            I_box(:, tt,bx) = I_box(:, tt,bx) + I_itp(strn, :)'; % 한 접속반의 전체 I-V curve
        end
    end
end

%%
figure;
%subplot(2,2,1)
for tt=1:timeea           
    plot(strV(strn,:,tt),I_std(:,tt),'Color',colors(tt,:),'LineWidth',0.7,'DisplayName',[num2str(tt+clock1st-1),'시']) % 합산된 스트링의 전압을 플롯    
    hold on;
end
legend('show')
grid on
xlabel('전압 (V)'); ylabel('전류 (A)')
title('각 접속반의 마지막 스트링의 I-V curve')

%% I-V curve 출력

figure;
%subplot(2,2,2)

for tt=1:timeea % 전제 시간에 대해 각 1번 접속반의 전체 I-V 커브를 플롯
    plot(V*srsn,I_box(:,tt,1),'Color',colors(tt,:),'LineWidth',0.7,'DisplayName',[num2str(tt+clock1st-1),'시']) 
    hold on
end
legend('show')
title('시간별 1번 접속반의 I-V curve')
xlabel('전압 (V)')
ylabel('전류 (A)')
grid on;
xlim([0,900])

%% P-V curve 출력

figure;
%subplot(2,2,3)

P_box=zeros(n,timeea,boxn); % 접속반의 P-V커브 제작을 위해 공간 사전 할당
for tt=1:timeea % 전체시간에 대해
    for bx=1:boxn % 접속반에 대해 
        P_box(:,tt,bx)=I_box(:,tt,bx).*V'*srsn; % 접속반 전류에 전압 곱해 전력 도출 (전류를 내삽해 얻기 위해 사용한 기준 전압이 V*srsn이었으므로 플롯도 같은 값에 대해 해야함)
    end
    plot(V'*srsn,P_box(:,tt,1)/1000,'Color',colors(tt,:),'LineWidth',0.7,'DisplayName',[num2str(tt+clock1st-1),'시']) % 모든 시간에서 1번 접속반의 전체 P-V curve 플롯
    hold on
end
title('시간별 1번 접속반 P-V curve')
xlabel('전압 (V)')
ylabel('전력 (kW)')
grid on
legend('show')
xlim([0,950])

%% 접속반 별 발전량 합산
P_inv=zeros(n,tt,invn); % 시간에 따른 각 인버터의 P-V curve
maxedP_inv=zeros(timeea,invn); % 시간에 따른 각 인버터의 최대 전력 (하나의 인버터에 하나의 MPPT 거침)
ivtdP_inv=zeros(timeea,invn); % 시간에 따른 각 인버터의 변환 교류전력

maxedP=zeros(1,timeea); % 시간에 따른 총 발전량
ivtdP=zeros(1,timeea); % 시간에 따른 총 변환 교류전력

for iv=1:invn % 인버터에 대해
    for tt=1:timeea % 모든 시간에 대해
        for sr=1:boxn/invn % 인버터 당 접속반에 대해
            P_inv(:,tt,iv)=P_inv(:,tt,iv)+P_box(:,tt,(iv-1)*boxn/invn+sr); % 접속반에 대해 합산해 인버터 별 P-V 곡선 도출
        end
        maxedP_inv(tt,iv)=max(P_inv(:,tt,iv)); % 인버터에 MPPT 적용 (단일 MPPT 방식)
        %fprintf("부하율 : %5.3f%%  ,  변환효율 : %5.3f%%\n",100*maxedP_inv(tt,iv)/invmax,efcv2(100*maxedP_inv(tt,iv)/invmax))
        % 변환 과정 중 오류가 많이 생겨 실시간 검증하기 위한 출력값
        ivtdP_inv(tt,iv)=maxedP_inv(tt,iv)*efcv2(100*maxedP_inv(tt,iv)/invmax)/100; % 앞에서 도출한 효율곡선을 사용해 손실 거친 변환값 계산
        maxedP(tt)=maxedP(tt)+maxedP_inv(tt,iv); % 모든 인버터의 입력 발전량을 합산
        ivtdP(tt)=ivtdP(tt)+ivtdP_inv(tt,iv); % 모든 인버터의 변환 발전량을 합산
    end    
end

tot_P=sum(maxedP); % 하루동안 생산된 발전량 계산
tot_ivtdP=sum(ivtdP); % 하루동안 변환된 발전량 계산

effP=100*ivtdP./maxedP; % 인버터 별로 달리 적용된 변환 효율의 평균
%effP(isnan(effP))=0;



%% dynamicInverter.m

prdP0=squeeze(predictP(pureG(1,1,:),maskT(1,1,:))*strn*srsn); % 순수 일사량을 통한 예측
prdP1=predictP(avgG,squeeze(maskT(1,1,:)))*strn*srsn; % 전체 일사량 평균을 통한 예측
prdP2=squeeze(predictP(avg_pureG',squeeze(maskT(1,1,:)))*strn*srsn*(1-prob_lv2)*(1-prob_lv3)).*cst_clound'*1.1; 
% 음영이 심한 패널을 제외한 일사량 평균에, 바이패싱으로 제외되는 패널의 개수만큼, 또한 구름으로 인해 제외되는 패널 개수만큼 상수
% 곱해준다.
% 부하율 상승으로 인해 입력 발전량이 더 클 것이므로 평가절상 보정계수 1.1 곱해줌


%%
tidx=[1:timeea]+clock1st-1; 
figure;
plot(tidx,prdP0/1000,'-b',tidx,prdP1/1000,'-g',tidx,prdP2/1000,'-r'); hold on;
legend('예측 발전량(ideal)','예측 발전량(평균)','예측 발전량(다이나믹)')
xlabel('시각 (시)'); ylabel('하루 총 발전량(kW)') 
title('모델 별 발전량 예측')
grid on;


%%



shdinvmax=100e3; % 음영패널 인버터 정a격전력

dinvn=3; % 다이나믹 인버터 개수
dboxn=6; % 다이나믹 접속반 개수
dbin=dboxn/dinvn; % 다이나믹 인버터 1기 당 접속반 개수


act_invn=zeros(1,timeea); %작동 중인 인버터 개수
dinvmax=300e3; % 다이나믹 인버터 정격전력
asgd_str=zeros(2,dboxn,timeea); % 할당된 스트링 위치



dstrV=zeros(strn,n,timeea); % 스트링 전압 (스트링별, 기준전류별, 시간별)

I_dbox=zeros(n,timeea,dboxn); % 접속반 전류 (시간별, 접속반별, 기준전압별)
P_box=zeros(n,timeea,boxn); % 접속반 별 전력

shdpn=zeros(1,timeea);
shdprln=zeros(1,timeea);
shdsrsn=zeros(1,timeea);

for tt=1:timeea
    tmp=isshd(:,:,tt);
    shdpn(tt)=sum(tmp(:)>=2); % 시간별 부분음영 패널 개수
    shdprln(tt)=ceil(sqrt(5*shdpn(tt)));
    shdsrsn(tt)=ceil(sqrt(shdpn(tt)/5));
end


shd_strV=zeros(max(shdprln),n,timeea);
I_shd=zeros(n,timeea); % 음영인버터 I-V 곡선
P_shd=zeros(n,timeea); % 음영인버터 P-V 곡선
dI_std=zeros(n,timeea);


for tt=1:timeea


    shd_count=1;
    act_invn(tt)=ceil(prdP2(tt)/dinvmax); % 작동할 인버터 대수 
    % (실제 입력 전력으로 하면 제일 정확하고 효율이 높겠지만, 실제 환경에선 고전압/전력 스위칭이 쉽지 않은 동작이므로 예측값에 의존하게 구현했다.)

    if act_invn(tt)>=4
        act_invn(tt)=3;
    end
    
    for iv=1:act_invn(tt) % 작동 중인 인버터에 대해
        for bx=1:dbin % 다이나믹 인버터 당 접속반에 대해
            asgd_str(:,dbin*(iv-1)+bx,tt)=[floor(strn*(dbin*(iv-1)+bx-1)/(act_invn(tt)*dbin))+1 , floor(strn*(dbin*(iv-1)+bx)/(act_invn(tt)*dbin))]; % 스트링 배정
        end
    end

    
    dI_std(:,tt)=linspace(0.001,max(Current(1,:,:,tt), [], 'all'),n); % 기준전류
    V_itp=zeros(srsn,n); % 18직렬에서 n개의 기준 전류에 의해 측정된 내삽 전압들
    I_itp = zeros(prln*boxn, n); % 17 스트링에 대한 n 개 기준 전압에서의 전류
    for st=1:strn
        for sr=1:srsn

        
            [unqItmp,unqidx]=unique(Current(:,st,sr,tt)); % 전류 중복값 제거
            unqVtmp=V(unqidx); % 중복값 제거된 전류와 같은 위상


            if length(unqItmp) > 1 && any(unqItmp > 0)
                V_itp(sr, :) = interp1(unqItmp, unqVtmp, dI_std(:,tt), 'linear'); % 내삽값
                V_itp(sr, isnan(V_itp(sr, :))) = 0; % NaN 제거
            else
                V_itp(sr, :) = zeros(1, n); % 유효한 데이터가 없으면 0으로 설정
            end

        if isshd(st,sr,tt)<=1 % 음영 없으면
            dstrV(st,:,tt)=dstrV(st,:,tt)+V_itp(sr,:); % 스트링 전압 : 직렬 전압 합산 
        else % 음영 있으면
            tmp=ceil(shd_count/shdsrsn(tt));
            shd_strV(tmp,:,tt)=shd_strV(tmp,:,tt)+V_itp(sr,:);
            shd_count=shd_count+1;
        end


        end
        
    end

%% 다이나믹 인버터 병렬
    for bx=1:dboxn
        for st=asgd_str(1,bx,tt):asgd_str(2,bx,tt)
            if st==0
                break
            end
            [unqVtmp, unqidx] = unique(dstrV(st, :,tt)); % 중복값 제거
            unqItmp = dI_std(unqidx,tt);
    
            % 보간하기 전에 데이터가 최소 두 개 이상인지 확인
            if length(unqVtmp) > 1 && any(unqVtmp > 0)
                I_itp(st, :) = interp1(unqVtmp, unqItmp, V*srsn, 'linear',0); % 보간 후 NaN 값은 0으로 처리
            else
                I_itp(st, :) = zeros(1, n); % 유효한 데이터가 없으면 0으로 설정
            end
    
            I_dbox(:, tt,bx) = I_dbox(:, tt,bx) + I_itp(st, :)';
        end
        P_box(:,tt,bx)=I_dbox(:,tt,bx).*V'*srsn;
    end  


%% 음영 인버터 병렬
    for pr=1:shdprln(tt)
   
        [unqVtmp, unqidx] = unique(shd_strV(pr, :,tt)); % 중복값 제거
        unqItmp = dI_std(unqidx,tt);

        % 보간하기 전에 데이터가 최소 두 개 이상인지 확인
        if length(unqVtmp) > 1 && any(unqVtmp > 0)
            I_itp(pr, :) = interp1(unqVtmp, unqItmp, V*max(shdsrsn), 'linear',0); % 보간 후 NaN 값은 0으로 처리
        else
            I_itp(pr, :) = zeros(1, n); % 유효한 데이터가 없으면 0으로 설정
        end
        
        I_shd(:, tt) = I_shd(:, tt) + I_itp(pr, :)';
    
        P_shd(:,tt)=I_shd(:,tt).*V'*max(shdsrsn);
    end  


end

%%

maxedP_box=zeros(timeea,dboxn); % 접속반 시간당 P_MPP
maxedP_inv=zeros(timeea,dinvn); % 인버터 시간당 P_MPP
ivtd_dP=zeros(timeea,dinvn);
ivtd_dP_time=zeros(1,timeea);

for bx=1:dboxn
    for tt=1:timeea
        maxedP_box(tt,bx)=maxedP_box(tt,bx)+max(P_box(:,tt,bx));
    end
    maxedP_inv(:,ceil(bx/dbin))=maxedP_inv(:,ceil(bx/dbin))+maxedP_box(:,bx);
    dP_time=sum(maxedP_inv,2);
    tot_dP=sum(dP_time);
end

for tt=1:timeea
    for iv=1:dinvn
        % fprintf("부하율 : %5.3f%%  ,  변환효율 : %5.3f%%\n",100*maxedP_inv(tt,iv)/dinvmax,efcv2(100*maxedP_inv(tt,iv)/dinvmax))
        ivtd_dP(tt,iv)=maxedP_inv(tt,iv)*efcv2(100*maxedP_inv(tt,iv)/dinvmax)/100;
    end 
    ivtd_dP_time=sum(ivtd_dP,2);
end
tot_ivtd_dP=sum(ivtd_dP_time);

maxedP_shd=max(P_shd); % 시간별 음영 P_MPP
tot_shdP=sum(maxedP_shd);
ivtd_shdP=zeros(1,timeea);
for tt=1:timeea
    ivtd_shdP(tt)=maxedP_shd(tt)*efcv2(100*maxedP_shd(tt)/shdinvmax)/100;
end
tot_ivtd_shdP=sum(ivtd_shdP);

effdP=100*ivtd_dP./maxedP_inv; % 다이나믹 인버터에서의 변환효율
effshdP=100*ivtd_shdP./maxedP_shd;

 
%% 3x3 figure plot

%subplot(3,3,1)
figure;
yyaxis left
bar(tidx,dP_time/1000,'FaceColor','#0D92F4'); hold on;
bar(tidx,ivtd_dP_time/1000,'FaceColor','#F95454'); hold on;
bar(tidx,(dP_time-ivtd_dP_time)/1000,'FaceColor','#C62E2E'); hold on;
plot(tidx,prdP2/1000,'-r'); hold on;
for tmp=1:max(act_invn)
    plot(tidx,tidx./tidx*dinvmax*tmp/1000,'--k','DisplayName',[num2str(tmp),'개 정격전력']);
    hold on;
end
xlabel('시각'); ylabel('발전량(kWh)') 
yyaxis right
for iv=1:dinvn
    plot(tidx,effdP(:,iv),'--o','DisplayName','변환효율')
end
ylabel('변환효율 (%)') 
legend('show')
title({'다이나믹 인버터 발전량', ['(직류 : ',num2str(tot_dP/1000),'kWh, 교류 : ',num2str(tot_ivtd_dP/1000),'kWh, 효율',num2str(tot_ivtd_dP/tot_dP*100),'%)']})
legend('입력 직류전력','변환 교류전력','변환 중 손실','예측 발전량(다이나믹)','1개 정격전력','2개 정격전력','3개 정격전력','변환효율')
grid on

%%

%subplot(3,3,2)
figure;
yyaxis left;
bar(tidx,maxedP_shd/1000,'FaceColor','#0D92F4'); hold on;
bar(tidx,ivtd_shdP/1000,'FaceColor','#F95454'); hold on;
bar(tidx,(maxedP_shd-ivtd_shdP)/1000,'FaceColor','#C62E2E'); hold on;
plot(tidx,tidx./tidx*shdinvmax/1000,'--k');
ylim([0,130])
title({'음영 인버터 발전량', ['(직류 : ',num2str(tot_shdP/1000),'kWh, 교류 : ',num2str(tot_ivtd_shdP/1000),'kWh, 효율',num2str(tot_ivtd_shdP/tot_shdP*100),'%)']})
xlabel('시각'); ylabel('발전량(kWh)') 

yyaxis right; grid on;
plot(tidx,effshdP,'--o')
xlabel('시각 (시)'); ylabel('변환효율 (%)')
ylim([0,100])
legend('입력 직류전력','변환 교류전력','변환 중 손실','예측 발전량(다이나믹)','정격전력','변환효율')

efftotdP=(ivtd_shdP+ivtd_dP_time')./(maxedP_shd+dP_time')*100;


%%

figure;
%subplot(3,3,3)
%subplot(1,2,1)
yyaxis left;
bar(tidx,(maxedP_shd'+dP_time)/1000,'FaceColor','#0D92F4'); hold on;
bar(tidx,(ivtd_shdP+ivtd_dP_time')/1000,'FaceColor','#F95454'); hold on;
bar(tidx,(maxedP_shd+dP_time'-ivtd_shdP-ivtd_dP_time')/1000,'FaceColor','#C62E2E'); hold on;
plot(tidx,prdP2/1000,'-r'); hold on;
for tmp=1:max(act_invn)
    plot(tidx,tidx./tidx*(dinvmax*tmp+shdinvmax)/1000,'--k','DisplayName',[num2str(tmp),'개 정격전력']);
    hold on;
end
title({'다이나믹 발전소 발전량', ['(직류 : ',num2str((tot_shdP+tot_dP)/1000),'kWh, 교류 : ',num2str((tot_ivtd_shdP+tot_ivtd_dP)/1000),'kWh, 효율',num2str((tot_ivtd_shdP+tot_ivtd_dP)/(tot_shdP+tot_dP)*100),'%)']})
xlabel('시각'); ylabel('발전량(kW)') 
grid on 
ylim([0,1100])

yyaxis right; grid on;
plot(tidx,effdP,'--o')
xlabel('시각 (시)'); ylabel('변환효율 (%)')
ylim([0,100])
legend('입력 직류전력','변환 교류전력','변환 중 손실','예측 발전량(다이나믹)','1+1개 정격전력','2+1개 정격전력','3+1개 정격전력','변환효율')


%%

figure;
%subplot(3,3,6)
%subplot(1,2,2)
yyaxis left; grid on;
bar(tidx,maxedP/1000,'FaceColor', [0.3, 0.7, 0.7]); hold on;
bar(tidx,ivtdP/1000,'FaceColor', [0.9, 0.7, 0.2]); hold on;
bar(tidx,(maxedP-ivtdP)/1000,'FaceColor',[0.8, 0.4, 0.4] ); hold on;
plot(tidx,prdP0/1000,'-b',tidx,prdP1/1000,'-g',tidx,prdP2/1000,'-r'); hold on;
plot(tidx,tidx./tidx*2*invmax/1000,'--k');
title({'센트럴 발전소 발전량', ['(직류 : ',num2str((tot_P)/1000),'kWh, 교류 : ',num2str((tot_ivtdP)/1000),'kWh, 효율',num2str((tot_ivtdP/tot_P*100)),'%)']})
xlabel('시각'); ylabel('하루 총 발전량(kW)') 

ylim([0,1100])

yyaxis right; grid on;
plot(tidx,effP,'--o')
xlabel('시각 (시)'); ylabel('변환효율 (%)')
ylim([0,100])
legend('입력 직류전력','변환 교류전력','변환 중 손실','예측 발전량(ideal)','예측 발전량(평균)','예측 발전량(다이나믹)','정격전력','변환효율')

%%

subplot(2,2,1)
grid on
for tt=1:timeea
    for st=1:strn/prln
        plot(dstrV(prln*st,:,tt),dI_std(:,tt),'Color',colors(tt,:),'LineWidth',0.7)
        hold on
    end
end
grid on
xlabel('전압 (V)'); ylabel('전류 (A)')
title('각 다이나믹 접속반 17th 스트링 I-V curve')



subplot(2,2,2)
for tt=1:timeea
    plot(V*max(shdsrsn),I_shd(:,tt),'Color',colors(tt,:),'LineWidth',0.7)
    hold on;
end
grid on
xlabel('전압 (V)'); ylabel('전류 (A)')
title('음영 인버터 I-V curve')





subplot(2,2,3)
for tt=1:timeea
    plot(V*srsn,P_box(:,tt,1)/1000,'Color',colors(tt,:),'LineWidth',0.7)
    hold on
end
title('시간별 1번 다이나믹 접속반 P-V curve')
xlabel('전압 (V)')    
ylabel('전력 (kW)')
grid on


subplot(2,2,4)
for tt=1:timeea
    plot(V*max(shdsrsn),P_shd(:,tt)/1000,'Color',colors(tt,:),'LineWidth',0.7,'DisplayName',[num2str(tt+clock1st-1),'시'])
    hold on;
end
title('시간별 1번 음영 인버터 P-V curve')
xlabel('전압 (V)')    
ylabel('전력 (kW)')
grid on
legend('show')


%%
figure;
%subplot(3,3,9)
bar(tidx,act_invn,'FaceColor','#89A8B2')
title('작동 다이나믹 인버터 개수')
xlabel('시각 (시)'); ylabel('인버터 개수 (개)') 
ylim([0,3.5])
grid on 



%% 결론


csmP_cnt=invmax*cst_csmP*ones(1,timeea)*2;
csmP_dyn=dinvmax*cst_csmP*act_invn.*ones(1,timeea);
csmP_shd=shdinvmax*cst_csmP*ones(1,timeea);




figure;

%subplot(2,2,1) % 각각 인버터 가동 개수, 가동 전력
plot(tidx,csmP_cnt/1000,tidx,csmP_dyn/1000,tidx,csmP_shd/1000)
legend('센트럴 인버터 가동 전력','다이나믹 인버터 가동 전력','음영 인버터 가동 전력')
xlabel('시각 (시)'); ylabel('kW')
title('인버터 별 가동 전력')

%%

%subplot(2,2,2) % 총합 비교
figure;
b=bar([  tot_P  ,tot_ivtdP-sum(csmP_cnt); tot_dP + tot_shdP  ,  tot_ivtd_dP+tot_ivtd_shdP-sum(csmP_shd)-sum(csmP_dyn)]/1000);
b(1).FaceColor = '#C6E7FF';
b(2).FaceColor = '#FFDDAE';
legend('입력 전력(DC)','유효 송전 전력(AC)') % Effective Delivered Power
 ylabel('발전량 (kWh)')
ylim([0.8*tot_P,1.1*tot_P]/1000)
grid on;

xticks(1:2); % x축 인덱스
xticklabels({'센트럴 인버터', '다이나믹 인버터'}); % 문자열 레이블
title('발전소 별 총 발전량 비교')


%% report illustration
vv=1:0.01:200;
yyaxis left
plot(vv,solarIV(1000,vv,25),'-','LineWidth',1.5,'Color','#4335A7'); hold on;
plot(vv,solarIV(600,vv,25),'-','LineWidth',1.5,'Color','#FCC737'); hold on;
plot(vv,solarIV(1000,vv,15),'-','LineWidth',1.5,'Color','#81BFDA'); hold on;
ylabel('전류 (A)')

yyaxis right
plot(vv,vv.*solarIV(1000,vv,25),'--','LineWidth',1.5,'Color','#4335A7'); hold on;
plot(vv,vv.*solarIV(600,vv,25),'--','LineWidth',1.5,'Color','#FCC737'); hold on;
plot(vv,vv.*solarIV(1000,vv,15),'--','LineWidth',1.5,'Color','#81BFDA'); hold on;

title('IV curve / PV curve')
legend('1000W/m^2, 25℃','600W/m^2, 25℃','1000W/m^2, 15℃')
xlabel('전압 (V)'); ylabel('전력 (W)')
xlim([1,55])
ylim([0,600])
%%

yyaxis left
plot(vv,solarIV(1000,vv,25),'-','LineWidth',1.5,'Color','#FF2929'); hold on; %패널1
plot(vv,solarIV(500,vv,25),'-','LineWidth',1.5,'Color','#3D3BF3'); hold on; %패널2
plot(vv,solarIV(900,vv,30),'-','LineWidth',1.5,'Color','#FA812F'); hold on; %패널3
plot(vv*2,solarIV(500,vv,25),'-','LineWidth',1.5,'Color','#FAB12F'); hold on; %패널1+2
plot(2*vv,solarIV(900,vv,30),'-','LineWidth',1.5,'Color','#7ED4AD'); hold on; %패널1+3
xlim([0,100])
xlabel('전압 (V)'); ylabel('전류 (A)')


yyaxis right
plot(vv,vv.*solarIV(1000,vv,25),'--','LineWidth',1.5,'Color','#FF2929'); hold on; %패널1
plot(2*vv,2*vv.*solarIV(900,vv,30),'--','LineWidth',1.5,'Color','#7ED4AD'); hold on; %패널1+3
plot(vv,vv.*solarIV(500,vv,25),'--','LineWidth',1.5,'Color','#3D3BF3'); hold on;
plot(vv*2,2*vv.*solarIV(500,vv,25),'--','LineWidth',1.5,'Color','#FAB12F'); hold on;
ylabel('전력 (W)')
legend('패널 1 (1000W/m^2, 25℃)','패널 2 (500W/m^2, 25℃)','패널 3 (900W/m^2, 30℃)','패널1 + 패널2 직렬연결','패널1 + 패널3 직렬연결')
ylim([0,1200])
grid on


%%
v0=1:0.1:60;
vmax=45.5199;
vv=1:0.01:vmax;
vv2=linspace(vmax,vmax*2,length(vv));
vv3=linspace(2*vmax,3*vmax,length(vv));

plot(vv,vv./vv+100,'-','LineWidth',1.5,'Color','#FF2929','DisplayName','패널 1 (1000W/m^2, 25℃)'); hold on;
plot(vv,vv./vv+100,'-','LineWidth',1.5,'Color','#9B7EBD','DisplayName','패널 2 (500W/m^2, 25℃)'); hold on;
plot(vv,vv./vv+100,'-','LineWidth',1.5,'Color','#608BC1','DisplayName','패널 3 (음영, 500W/m^2, 25℃)'); hold on;
plot(vv,vv./vv+100,'-','LineWidth',1.5,'Color','#FAB12F','DisplayName','패널 1 + 2 + 3 직렬연결 (비 바이패스)'); hold on;
plot(vv,vv./vv+100,'-','LineWidth',1.5,'Color','#7ED4AD','DisplayName','패널 1 + 2 + 3 직렬연결 (바이패스)'); hold on;

plot(v0-0.5,solarIV(1000,v0,25)-0.07,'-','LineWidth',1.5,'Color','#FF2929','HandleVisibility','off'); hold on; %패널1
plot(v0-0.9,solarIV(1000,v0,25)-0.14,'-','LineWidth',1.5,'Color','#9B7EBD','HandleVisibility','off'); hold on; %패널2
plot(vv,vv./vv*5.263,'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 초반
plot(vv2,vv2./vv2*5.263,'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 중반
plot(vv3-0.5,solarIV(500,vv,25)-0.07,'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 후반
plot(v0,solarIV(500,v0,25),'-','LineWidth',1.5,'Color','#608BC1','HandleVisibility','off'); hold on; %패널3 음영
good=solarIV(1000,vv,25);
bad=solarIV(500,vv,25);
plot(vv,vv./vv*10.6653,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); hold on; % 비 바이패스 초반
plot(vv2,good,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); hold on; % 바이패스 전반
plot(vv3,bad,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); % 바이패스 후반
axis([1,150,0,13])
grid on;
xlabel('전압 (V)'); ylabel('전류 (A)')
legend('show');
title('바이패스 다이오드 여부에 따른 직렬 I-V 곡선')

%%

%%

plot(vv,vv./vv-100,'-','LineWidth',1.5,'Color','#FF2929','DisplayName','패널 1 (1000W/m^2, 25℃)'); hold on;
plot(vv,vv./vv-100,'-','LineWidth',1.5,'Color','#9B7EBD','DisplayName','패널 2 (500W/m^2, 25℃)'); hold on;
plot(vv,vv./vv-100,'-','LineWidth',1.5,'Color','#608BC1','DisplayName','패널 3 (음영, 500W/m^2, 25℃)'); hold on;
plot(vv,vv./vv-100,'-','LineWidth',1.5,'Color','#FAB12F','DisplayName','패널 1 + 2 + 3 직렬연결 (비 바이패스)'); hold on;
plot(vv,vv./vv-100,'-','LineWidth',1.5,'Color','#7ED4AD','DisplayName','패널 1 + 2 + 3 직렬연결 (바이패스)'); hold on;

plot(v0-0.5,(v0-0.5).*(solarIV(1000,v0,25)-0.07),'-','LineWidth',1.5,'Color','#FF2929','HandleVisibility','off'); hold on; %패널1
plot(v0-0.9,(v0-0.9).*(solarIV(1000,v0,25)-0.14),'-','LineWidth',1.5,'Color','#9B7EBD','HandleVisibility','off'); hold on; %패널2
plot(vv,vv*5.263,'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 초반
plot(vv2,vv2*5.263,'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 중반
plot(vv3-0.5,(vv3-0.5).*(solarIV(500,vv,25)-0.07),'-','LineWidth',1.5,'Color','#FAB12F','HandleVisibility','off'); hold on; % 비 바이패스 후반
plot(v0,v0.*solarIV(500,v0,25),'-','LineWidth',1.5,'Color','#608BC1','HandleVisibility','off'); hold on; %패널3 음영
good=solarIV(1000,vv,25);
bad=solarIV(500,vv,25);
plot(vv,vv*10.6653,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); hold on; % 비 바이패스 초반
plot(vv2,vv2.*good,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); hold on; % 바이패스 전반
plot(vv3,vv3.*bad,'-','LineWidth',1.5,'Color','#7ED4AD','HandleVisibility','off'); % 바이패스 후반
axis([1,150,0,900])
grid on;
xlabel('전압 (V)'); ylabel('전력 (W)')
legend('show');
title('바이패스 다이오드 여부에 따른 직렬 P-V 곡선')



%% functions


function output = efcv2(x)
    % 결과를 초기화 (모두 NaN으로)
  
    % 각 조건에 맞는 인덱스
    idx1 = (x >= 0) & (x < 2.68);
    idx2 = (x >= 2.68);
    %%idx3 = (x >= 39) & (x <= 100);

    % 각 조건에 맞는 계산 수행
    output(idx1) = 13.07 * x(idx1)+19.53;
    output(idx2) = -130./x(idx2)+100;
    %%output(idx3) = -0.0231 * x(idx3) + 99.314;
end


function output=predictP(G,T)
    cst_b=-1;% 온도계수 (%/K)
    P_mr=305;% 기준 전력 (W)
    G_r=1000;% 기준 일사량 (W/m2)
    T_r=25;% 기준 온도 (℃)
    
    res = P_mr * G / G_r .* ( 1 + ( cst_b /100 ) * ( T - T_r ));
    res(res<0)=0;
    output=res;
end



function [I,Pss]=solarIV(G,V,T)

    k_s=-1e-2; %%일사량, 패널 온도
    N_s=144;%%셀 개수
    i_sc=10.69; %%단락전류 집인태양광 10.69
    G_0=1000; %%기준일사량
    C_t=0.00247; %%광전류 온도계수 ㅇㅇㅇㅇ
    T_ref=25+273; %%기준온도
    i_s0=2.7e-5; %%포화전류 ㄴㄴㄴ 이거 줄이면 voc늘어남
    qk=1.16e4; %%전자 전하량
    A=1;
    %%k=1.38e-23; %%볼츠만 상수
    R_sh=1000; %%병렬저항
    
    T_s = T+273 + k_s * G; %%일사량 영향 패널 온도 (행렬)
    
    E_g=(1.02-20*(T_s-T_ref))*1.6e-3; %%에너지갬
    
    v_d = V / N_s; %%셀 당 전압 (행렬)
    i_ph = i_sc * G / G_0 + C_t * ( T_s - T_ref ); %%생성전류 (행렬)
    i_0 = i_s0 * ( T_s / T_ref ).^3 .* exp( qk * E_g / ( A ) .* ( 1/T_ref - 1./T_s )); %%포화전류(다이오드누설) (행렬)
    i_d = i_0 .* ( exp( qk * v_d ./ ( A * T_s )) -1 ); %%다이오드 전류? (행렬)
    i_r = v_d / R_sh; %%병렬전류? (행렬)
    
    I = i_ph - i_d - i_r;
    I(I<0)=0;
    
    
    Pss=V.*I;
    Pss(Pss<0)=0;

end
