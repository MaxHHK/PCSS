classdef CPCSS1 < PROBLEM
% <multi> <real> <large/none> <constrained> <expensive/none>
% The time-varying ratio error estimation problem
% T --- 1000 --- Length of data (related to the number of variables)

%------------------------------- Reference --------------------------------
% C. He, R. Cheng, C. Zhang, Y. Tian, Q. Chen, and X. Yao, Evolutionary
% large-scale multiobjective optimization for ratio error estimation of
% voltage transformers, IEEE Transactions on Evolutionary Computation,
% 2020, 24(5): 868-881.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(Access = private)
        pc_num;
        K = 24 * 60; % One day, 24 hours, control every minutes.
        Mean;   % Mean values of the dataset
        capacity;
        required;
        consume;
        times;
        schedule;
        offschedule;
        costs;
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            % Load data
            obj.pc_num = obj.D; % Number of pneumatic conveying systems
            CallStack = dbstack('-completenames');
            load(fullfile(fileparts(CallStack(1).file),'Dataset_PCSS.mat'),'PCSS1');
            % Set the numbers of objectives and decision variables
            obj.M = 2;
            obj.D = obj.pc_num*obj.K;
            % The decision variables are the offset of the mean values of
            % the dataset
            obj.lower    = zeros(1, obj.D);
            obj.upper    = ones(1, obj.D);
            obj.encoding = ones(1,obj.D) * 4;
            %
            obj.capacity = PCSS1.capacity(1:obj.pc_num);
            obj.consume  = PCSS1.consumption(1:obj.pc_num);
            obj.required = PCSS1.required(1:obj.pc_num);
            obj.times = ceil(obj.required ./ obj.capacity);
            obj.costs = PCSS1.costs;
            for i = 1:obj.pc_num
                obj.schedule{i} = ones(1, obj.times(i));
                obj.offschedule{i} = zeros(1, obj.K - obj.times(i));
            end
        end
        %% Generate initial solutions
        function Population = Initialization(obj,N)
            if nargin < 2; N = obj.N; end
            pop_sche = {};
            PopDec = zeros(N, obj.D);
            for i_n = 1:obj.N
                all_sche = zeros(obj.pc_num, obj.K);
                for i_pc = 1:obj.pc_num
                    sche = [obj.schedule{i_pc}, obj.offschedule{i_pc}];
                    randIndex = randperm(obj.K);
                    sche = sche(randIndex);
                    all_sche(i_pc, :) = sche;
                end
                pop_sche{i_n, 1} = all_sche;
                PopDec(i_n, :) = reshape(all_sche', obj.D, 1)';
            end
            
            PopDec = logical(randi([0,1],N,obj.D));
            Population = obj.Evaluation(PopDec);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            N      = size(PopDec,1);
            PopObj = zeros(N,obj.M);
            for i_pop = 1:N
                raw_sche = PopDec(i_pop, :);
                sche = reshape(raw_sche, obj.K, obj.pc_num)';
                run_time = sum(sche');
                differ = run_time - obj.times;
                op1 = sum(differ(differ>=0));

                consume_per = obj.consume' .* sche;
                consume_sum = sum(consume_per);
                fluc = 0;
                for k = 2:obj.K
                    fluc = fluc + abs(consume_sum(k) - consume_sum(k-1));
                end

                capacity_per = obj.capacity' .* sche;
                capacity_sum = sum(capacity_per);
                cost_sum = sum(obj.costs .* capacity_sum);
                
                PopObj(i_pop, 1) = op1;
                PopObj(i_pop, 2) = fluc;
                if obj.M == 3
                    PopObj(i_pop, 3) = cost_sum;
                end
            end
        end
        %% Calculate constraint violations
        function PopCon = CalCon(obj,PopDec)
            N      = size(PopDec,1);
            PopCon = zeros(N,1);
            for i_pop = 1:N
                raw_sche = PopDec(i_pop, :);
                sche = reshape(raw_sche, obj.K, obj.pc_num)';
                run_time = sum(sche');
                differ = run_time - obj.times;
                PopCon(i_pop) = sum(differ<0);
            end
        end
        %% Generate a point for hypervolume calculation
        function R = GetOptimum(obj,~)
            X = zeros(1,obj.D);
            X(1:2:end) = obj.lower(1:2:end);
            X(2:2:end) = obj.upper(2:2:end);
            R = obj.CalObj(X);
            R = zeros(1, obj.M);
        end
    end
end