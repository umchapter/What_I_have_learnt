import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 단어 지정하고, ctrl + d : 한 번에 여러 개 수정 
# ctrl + ] : 오른쪽으로 인덴트 이동
df = pd.read_csv("./datasets/tmdb_5000_movies.csv")
df_2 = df.reset_index(drop=True)

df_2["genres"] = df_2["genres"].apply(literal_eval)
df_2["keywords"] = df_2["keywords"].apply(literal_eval)

df_2["genres"] = df_2["genres"].apply(lambda x : [y["name"] for y in x])
df_2["keywords"] = df_2["keywords"].apply(lambda x : [y["name"] for y in x])

df_2["genres_literal"] = df_2["genres"].apply(lambda x : (" ").join(x))

# 상위 60%에 해당하는 vote_count를 최소 투표 횟수인 m으로 지정
C = df_2["vote_average"].mean()   # 약 6점
m = df_2["vote_count"].quantile(0.6)  # 370회

# 가중평점을 계산하는 함수
def weighted_vote_average(record) :
    v = record["vote_count"]
    R = record["vote_average"]

    return ((v/(v+m))*R + (m/(m+v))*C)  # 가중평균을 return

def find_sim_movie_ver2(df, sorted_ind, title_name, top_n=10) :
    title_moive = df[df['title'] == title_name]
    title_index = title_moive.index.values

    # top_n의 2배에 해당하는 장르 유사성이 높은 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]


class Recommand() :
    def __init__(self, vectorizer) :
        self.vectorizer = vectorizer

    def get_recommandation(self, title, dataframe=df_2) :

        vecorized = self.vectorizer(min_df=0, ngram_range=(1,2))
        genre_mat = vecorized.fit_transform(dataframe["genres_literal"])
        genre_sim = cosine_similarity(genre_mat, genre_mat)
        genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]

        # 기존 데이터에 가중평점 칼럼 추가
        dataframe["weighted_vote"] = dataframe.apply(weighted_vote_average, axis=1)
        similar_movies = find_sim_movie_ver2(dataframe, genre_sim_sorted_ind, title, 10)

        result_df = pd.DataFrame(columns=["title", "date"])
        result_df["title"] = similar_movies["title"]
        result_df["date"] = similar_movies["release_date"]

        # movie_names = []
        # for i in similar_movies["title"] :
        #     movie_names.append(i)
        # print(movie_names)
        return result_df

# recommander = Recommand(TfidfVectorizer)

# recommander.get_recommandation("The Shawshank Redemption")

# print(similar_movies[["title", "vote_average", "weighted_vote", "genres", "vote_count"]])