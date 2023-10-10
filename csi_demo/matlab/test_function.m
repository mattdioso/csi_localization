function test = test_function(mat_file)
%disp(mat_file);
data = load('-ascii', mat_file);
SN = unique (sort (data (:, 1)));
test = []
end
