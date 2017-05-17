% generate plots

fId = fopen( 'save.log' );
%fgetl( fId ); % skip line
tempData = textscan( fId, '%s %d %f %f ', 'Delimiter', ',' );

rewards = zeros(10,51);
gens = zeros(10,51);
currMetaGen = 0;

cand = 0
for i = 1:size( tempData{1,1}, 1 )
    if strcmp( tempData{1,1}(i), 'meta-main' )
        currMetaGen = tempData{1,2}(i);
    elseif strcmp( tempData{1,1}(i), 'meta-cand' )
        cand = cand+1;
        %rewards( size(rewards,1) + 1 ) = tempData{1,3}(i); 
    elseif strcmp( tempData{1,1}(i), 'es' )
        pos = tempData{1,2}(i);
        rewards( pos + 1, cand ) = tempData{1,3}(i);
        gens( pos + 1,cand ) = currMetaGen + tempData{1,2}(i);
    end
end

figure
for i = 1:cand-1
    plot( gens(:, i), rewards(:,i) )
    hold on
end
title( 'ES for meta-learning')
xlabel('Generation')
ylabel('Reward')

fId = fopen( 'gridsearch.log' );
%fgetl( fId ); % skip line
tempData = textscan( fId, '%s %d %f %f ', 'Delimiter', ',' );

rewards = zeros(8,50);
gens = zeros(8,50);
currMetaGen = 0;

cand = 0;
for i = 1:size( tempData{1,1}, 1 )
    if strcmp( tempData{1,1}(i), 'meta-main' )
        currMetaGen = tempData{1,2}(i);
    elseif strcmp( tempData{1,1}(i), 'meta-cand' )
        %rewards( size(rewards,1) + 1 ) = tempData{1,3}(i); 
    elseif strcmp( tempData{1,1}(i), 'es' )
        pos = tempData{1,2}(i);
        if pos == 0
            cand = cand + 1;
        end
        rewards( cand, pos + 1 ) = tempData{1,3}(i);
        gens( cand, pos +1 ) = pos;
    end
end

figure
for i = 1:cand
    plot( gens(i, :), rewards(i,:) )
    hold on
end
title( 'Grid search for meta-learning')
xlabel('Generation')
ylabel('Reward')