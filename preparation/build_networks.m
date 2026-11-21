function build_networks(catRoot, spmDir, preparedDir, ref)
% Reconstruct the five-feature bilateral networks used in the submitted study.
% Each subject's CAT12 central surfaces, registered spheres and thickness files
% are read once. Locally saved matrices and parcel strengths use the same order.
addpath(spmDir);
catDir = fullfile(spmDir,'toolbox','cat12');
addpath(catDir);
if nargin < 4
    ref = fullfile(preparedDir,'reference');
end
cohort = readtable(fullfile(preparedDir,'cohort.csv'),'TextType','string','VariableNamingRule','preserve');
R = [400,600];
for side = 1:2
    hemi = ["lh","rh"];
    sphere = gifti(char(fullfile(catDir,'templates_surfaces',hemi(side)+".sphere.freesurfer.gii")));
    xyz = double(sphere.vertices);
    tree{side} = KDTreeSearcher(xyz./vecnorm(xyz,2,2));
end
for a = 1:2
    folder = fullfile(ref,"schaefer"+R(a));
    labels = load(fullfile(folder,'matlab_labels.mat'),'labels');
    parcels{a} = double(labels.labels(:))+1;
    names{a} = strip(readlines(fullfile(folder,'regions.txt')));
    names{a} = names{a}(names{a}~="");
    strength{a} = zeros(height(cohort),R(a));
    assert(numel(names{a})==R(a));
    mkdir(fullfile(preparedDir,"schaefer"+R(a)+"_individual"));
end
for i = 1:height(cohort)
    features = {[],[]}; ids = {[],[]};
    anat = fullfile(catRoot,cohort.participant_id(i),'anat');
    for side = 1:2
        h = hemi(side);
        central = gifti(char(fullfile(anat,h+".central.T1.gii")));
        sphere = gifti(char(fullfile(anat,h+".sphere.reg.T1.gii")));
        xyz = double(sphere.vertices);
        idx = knnsearch(tree{side},xyz./vecnorm(xyz,2,2));
        thickness = double(cat_io_FreeSurfer('read_surf_data',char(fullfile(anat,h+".thickness.T1"))));
        thickness = thickness(:);
        [area,mc,gc] = vertex_geometry(double(central.vertices),double(central.faces));
        f = [thickness,area,thickness.*area,mc,gc];
        assert(all(isfinite(f),'all'));
        for a = 1:2
            features{a} = [features{a}; f];
            ids{a} = [ids{a};parcels{a}((side-1)*163842+idx)];
        end
    end
    for a = 1:2
        n = R(a); f = features{a}; label = ids{a};
        regional = zeros(n,5); vertexCounts = zeros(n,1);
        for j = 1:n
            values = f(label==j,:); vertexCounts(j)=size(values,1);
            assert(vertexCounts(j)>0);
            regional(j,:) = [mean(values(:,1)),sum(values(:,2)),sum(values(:,3)),mean(values(:,4)),mean(values(:,5))];
        end
        standardized = (regional-mean(regional,1))./std(regional,0,1);
        network = corr(standardized','Type','Pearson','Rows','complete');
        network(1:n+1:end) = 0;
        strength{a}(i,:) = sum(network,2)'/(n-1);
        save(fullfile(preparedDir,"schaefer"+n+"_individual",cohort.participant_id(i)+".mat"),'network','regional','vertexCounts');
    end
end
for a = 1:2
    t = array2table(strength{a},'VariableNames',cellstr(names{a}));
    t = addvars(t,cohort.participant_id,'Before',1,'NewVariableNames','Subject');
    writetable(t,fullfile(preparedDir,"schaefer"+R(a)+"_nodal60.csv"));
end
end
