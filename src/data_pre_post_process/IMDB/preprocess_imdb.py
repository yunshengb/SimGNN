import sys
from os.path import dirname, abspath

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../'.format(cur_folder))

from utils import get_data_path, create_dir_if_not_exists, save_as_dict, load_as_dict
from collections import OrderedDict
import networkx as nx
import csv
import sys
import random


class Movie(object):
    def __init__(self, mid, num_votes):
        self.mid = mid
        self.num_votes = num_votes
        self.people = OrderedDict()
        self.type = None
        self.title = None
        self.rank = None

    def add_person(self, pid, ordering, job):
        self.people[pid] = (ordering, job)

    def set_type(self, type):
        self.type = type

    def set_title(self, title):
        self.title = title

    def set_rank(self, rank):  # zero-based, so that can be used as graph id
        self.rank = rank

    def remove_one_person(self):
        l = len(self.people)
        pid_to_remove = None
        largest_ordering = -float('inf')
        for pid in self.people.keys():
            ordering = self.people[pid][0]
            if ordering > largest_ordering:
                pid_to_remove = pid
                largest_ordering = ordering
        del self.people[pid_to_remove]
        print('{} removed'.format(largest_ordering))
        assert(len(self.people) == l - 1)


class Person(object):
    def __init__(self, pid, name):
        self.pid = pid
        self.name = name
        self.movies = []
        self.chosen = False

    def add_movie(self, mid):
        self.movies.append(mid)

    def set_chosen(self):
        self.chosen = True


def main():
    sfn = cur_folder + '/temp'
    loaded = load_as_dict(sfn)
    if not loaded:
        movies, movies_dict, people_dict = read_data()
        print("finish reading data!")
        movies.sort(key=voteGetter, reverse=True)
        print('sorted')
        for idx, movie in enumerate(movies):
            movie.set_rank(idx)
        print('shuffled')
        save_as_dict(sfn, movies, movies_dict, people_dict)
    else:
        movies = loaded['movies']
        movies_dict = loaded['movies_dict']
        people_dict = loaded['people_dict']
        print('loaded movies, movies_dict, people_dict')

    create_dataset(movies, movies_dict, people_dict, 'Coarse')
    create_dataset(movies, movies_dict, people_dict, 'Fine')


def read_data():
    # max limit size
    csv.field_size_limit(sys.maxsize)
    print('start loading data')

    movies = []
    movies_dict = {}
    with open(r'ratings_data.tsv') as rating_data_tsvfile:
        rating_data_TSV = csv.reader(rating_data_tsvfile, delimiter='\t')
        # ['tt0000001', '4.5', '1000']
        for row in rating_data_TSV:
            if row[2] != 'numVotes' and int(row[2]) > 10000: # basic threshold
                num_votes = int(row[2])
                mid = row[0]
                movie = Movie(mid, num_votes)
                movies.append(movie)
                movies_dict[mid] = movie
    print('loaded ratings_data', len(movies))

    people_dict = {}
    with open(r'name_data.tsv') as name_data_tsvfile:
        name_date_TSV = csv.reader(name_data_tsvfile, delimiter='\t')
        # ['nm0278768', 'Tom Fiscella', '\\N', '\\N', 'actor', 'tt0306673,tt0479095,tt0278801,tt0285331']
        for row in name_date_TSV:
            pid = row[0]
            people_dict[pid] = Person(pid, row[1])
    print('loaded name_data', len(people_dict))

    with open(r'principals_data.tsv') as principals_data_tsvfile:
        principals_data_TSV = csv.reader(principals_data_tsvfile,
                                         delimiter='\t')
        # ['tt0075129', '10', 'nm0302976', 'editor', '\\N', '\\N']
        for row in principals_data_TSV:
            mid = row[0]
            if mid not in movies_dict:
                continue
            movie = movies_dict[mid]
            pid = row[2]
            if not pid in people_dict:
                continue
            person = people_dict[row[2]]
            person.set_chosen()
            person.add_movie(movie.mid)
            ordering = int(row[1])
            movie.add_person(pid, ordering, row[3])
    print('loaded principals_data')

    for pid in list(people_dict.keys()):
        person = people_dict[pid]
        if not person.chosen:
            del people_dict[pid]
    print('after iterating through movies, {} people'.format(len(people_dict)))

    with open(r'title_data.tsv') as title_data_tsvfile:
        title_data_TSV = csv.reader(title_data_tsvfile, delimiter='\t')
        # ['tt0279967', 'video', 'Mulan 2', 'Mulan II', '0', '2004', '\\N', '79', 'Action,Animation,Comedy']
        for row in title_data_TSV:
            mid = row[0]
            if mid not in movies_dict:
                continue
            movie = movies_dict[mid]
            movie.set_type(row[1])
            movie.set_title(row[2])
    print('loaded title_data')

    return movies, movies_dict, people_dict


def create_dataset(movies, movies_dict, people_dict, fc):
    assert (fc == 'Fine' or fc == 'Coarse')
    dirout = get_data_path() + '/IMDB1k{}'.format(fc)
    dirout_train = dirout + '/train'
    dirout_test = dirout + '/test'
    create_dir_if_not_exists(dirout_train)
    create_dir_if_not_exists(dirout_test)

    graphs = []
    for movie in movies:
        mid = movie.mid
        num_nodes = 1 + len(movie.people)
        if num_nodes >= 9:
            for _ in range(num_nodes - 9):
                movie.remove_one_person()
            assert (len(movie.people) == 8)
            num_nodes = 9
        g = nx.Graph()
        if fc == 'Coarse':
            g.add_node(0, type=movie.type)
            if movie.type != 'movie':
                print(movie.type)
        else:
            g.add_node(0, name=movie.title)
        # Construct people nodes and the links between the movie and its people.
        nid = 1
        for pid, (ordering, job) in movie.people.items():
            if fc == 'Coarse':
                g.add_node(nid, type=job)
            else:
                g.add_node(nid, name=people_dict[pid].name)
            g.add_edge(0, nid)
            nid += 1
        # Construct the collaboration links between the people.
        pid_list = list(movie.people.keys())
        for i in range(len(pid_list)):
            for j in range(i + 1, len(pid_list)):
                pid1 = pid_list[i]
                pid2 = pid_list[j]
                nid1 = i + 1
                nid2 = j + 1
                person1 = people_dict[pid1]
                person2 = people_dict[pid2]
                collab = list(set(person1.movies) & set(person2.movies))
                assert (mid in collab)
                if len(collab) >= 2:
                    # They have at least one another movie they collaborated on.
                    g.add_edge(nid1, nid2)
        g.graph['gid'] = movie.rank
        graphs.append(g)
        assert (g.number_of_nodes() == num_nodes)
        print('Done {} with {} nodes {} edges'.format(movie.title, num_nodes, g.number_of_edges()))

    print('Loaded {} graphs'.format(len(graphs)))

    graphs = graphs[0:1000]

    random.Random(123).shuffle(graphs)

    for g in graphs[0:800]:
        nx.write_gexf(g, dirout_train + '/{}.gexf'.format(g.graph['gid']))
    for g in graphs[800:1000]:
        nx.write_gexf(g, dirout_test + '/{}.gexf'.format(g.graph['gid']))


def voteGetter(elem):
    return elem.num_votes


if __name__ == '__main__':
    main()
