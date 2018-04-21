function RemMorphs

    
    % REM parameters
    
    N=20;   % number of features per item
    u=.1;  % encoding probabilty
    g=.2;  % geometric parameter
    c=.7;  % probability of accuracy encoding
    
    length=28; % list length
    n_sim = 1000; % number of simulations to average over
    
    % calculate log likelihood values for features

    MaxValue=100;  % if g<.1, then need to choose a higher max value for this to ensure approximately 100% for the CDF
    LogLMatch=zeros(MaxValue,1);
    for i=1:MaxValue  
       LogLMatch(i)=c+(1-c)*g*((1-g)^(i-1));
       LogLMatch(i)=log(LogLMatch(i)/(g*((1-g)^(i-1))));
    end
    LogLMismatch=log(1-c);
   
    for S=1:15;  % step through different numbers of study attempts (S)
        for i=1:n_sim   % step through simulated lists

            targets=geornd(g,length,N);   % create targets

            distractors=geornd(g,length,N);  % create distractors

            for j=1:2:length            % create morphs using first half from odd items and second half from even items
                TargMorph((j+1)/2,[1:N/2])=targets(j,[1:N/2]);  
                TargMorph((j+1)/2,[1+N/2:N])=targets(j+1,[1+N/2:N]);
                DistMorph((j+1)/2,[1:N/2])=distractors(j,[1:N/2]);  
                DistMorph((j+1)/2,[1+N/2:N])=distractors(j+1,[1+N/2:N]);
            end

            encode = rand(length,1) < (1 - ((1 - u)^S));  % vector of which trials were encoded
            encode = encode *ones(1,N);      % expand trial encoding to full trial by feature matrix

            accurate=rand(length,N)<c;   % determine whether storage is accurate

            CorrectStore=logical(encode.*accurate);  % matrices for correct and error storage
            ErrorStore=logical(encode.*(1-accurate));
            
            % populate memory matrix (M)
            M=-ones(length,N);  % -1 is used to indicate a lack of encoding
            M(CorrectStore)=targets(CorrectStore);
            M(ErrorStore)=geornd(g,sum(sum(ErrorStore==1)),1); % for error storage, sample new geometric values

            for j=1:length   % step through test list with equal number of targets and distractors
                TargOdds(i,j)=CalcOdds(targets(j,:));
                DistOdds(i,j)=CalcOdds(distractors(j,:));
            end
            for j=1:length/2  % step through test list for target and distractor morphs
                TargMorphOdds(i,j)=CalcOdds(TargMorph(j,:));
                DistMorphOdds(i,j)=CalcOdds(DistMorph(j,:));
            end
        end

        C=1;  % default criterion of equal odds

        HitRate(S)=mean(mean(TargOdds>C))+mean(mean(TargOdds==C))/2   % use default criterion of 1
        FaRate(S)=mean(mean(DistOdds>C))+mean(mean(DistOdds==C))/2
        TarMorphRate(S)=mean(mean(TargMorphOdds>C))+mean(mean(TargMorphOdds==C))/2   % flip a coin for values equal to the criterion
        DistMorphRate(S)=mean(mean(DistMorphOdds>C))+mean(mean(DistMorphOdds==C))/2
    end
    
    figure(1);
    hold off
    plot(HitRate','-k');
    hold on
    plot(TarMorphRate','-r');

    
    function Odds=CalcOdds(item)  % pass in a test item and calculate odds
        
        RepItem=(ones(length,1)*item);  % create matrix based on test item that matches size of memory
        matches=RepItem==M;             % matrix of matches
        mismatches=(M>-1).*(1-matches);  % mismatches are only for stored features
        
        FamMismatch=LogLMismatch.*sum(mismatches,2);  % sum mismatch log likelihoods for each item in memory
        
        MatchItem=LogLMatch(item+1)';     % plus one because Matlab geometric starts at 0
        RepMatch=(ones(length,1)*MatchItem);  % matrix of log likelihood values if there was a match
        FamMatch=matches.*RepMatch; % matrix of log lilelihood vales only for matches
        FamMatch=sum(FamMatch,2);  % sum match log likelihoods for each item in memory

        LogL=FamMismatch+FamMatch;  % sum matching and mismatching log likelihoods
        Odds=sum(exp(LogL))./length; % convert to likelihood, sum, and divide by number of items in memory
    end

end

   