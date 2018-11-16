from utils import exec, get_model_path

exp = 'siamese_regression_linux_2018-08-07T01:32:40'

# imdb1kcoarse_2018-07-28T10:22:27
# aids700nef_2018-07-28T10:14:20
# linux_2018-07-28T10:14:56
# imdb1kcoarse_2018-07-28T10:15:00
# aids700nef_2018-07-28T10:16:44
# linux_2018-07-28T10:16:57
# imdb1kcoarse_2018-07-28T10:17:03
# aids700nef_2018-07-28T10:24:51
# linux_2018-07-28T10:25:36
# imdb1kcoarse_2018-07-28T10:25:39

exec('scp -r yba@qilin.cs.ucla.edu:/home/yba/GraphEmbedding/model/Siamese/logs/{} '
     '{}/Siamese/logs'.format(exp, get_model_path()))
print('done')
