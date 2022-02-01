---
layout: single
title: "[Recommendation System] TMDB 5000 데이터를 이용한 콘텐츠 기반 필터링 실습"
categories: ['AI', 'RecommendationSystem']
---



<br>

# TMDB 5000 데이터를 이용한 콘텐츠 기반 필터링 실습

TMDB 5000은 캐글의 영화 데이터 세트입니다. 


```python
# 필요 라이브러리 import
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
```

<br>

### 1. 데이터 전처리


```python
from ast import literal_eval

movies =pd.read_csv('assets/tmdb_5000_movies.csv')
movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                    'popularity', 'keywords', 'overview']]
pd.set_option('max_colwidth', 100)
movies_df[['genres','keywords']][:1]

movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)

movies_df['genres'] = movies_df['genres'].apply(lambda x : [ y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [ y['name'] for y in x])

movies_df[['genres', 'keywords']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Action, Adventure, Fantasy, Science Fiction]</td>
      <td>[culture clash, future, space war, space colony, society, space travel, futuristic, romance, spa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Adventure, Fantasy, Action]</td>
      <td>[ocean, drug abuse, exotic island, east india trading company, love of one's life, traitor, ship...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Action, Adventure, Crime]</td>
      <td>[spy, based on novel, secret agent, sequel, mi6, british secret service, united kingdom]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Action, Crime, Drama, Thriller]</td>
      <td>[dc comics, crime fighter, terrorist, secret identity, burglar, hostage drama, time bomb, gotham...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Action, Adventure, Science Fiction]</td>
      <td>[based on novel, mars, medallion, space travel, princess, alien, steampunk, martian, escape, edg...</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환 
from sklearn.feature_extraction.text import CountVectorizer

# 딕셔너리 형태를 리스트로 변환한 genres_literal 칼럼 생성
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
```


```python
from sklearn.metrics.pairwise import cosine_similarity

# 코사인 유사도
genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
# 장르 유사도 리스트
print(genre_sim_sorted_ind[:1])
```

    [[   0 3494  813 ... 3038 3037 2401]]

<br>

### 2. 코사인 유사도가 높은 영화 검색


```python
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    
    # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
    title_movie = df[df['title'] == title_name]
    
    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    
    # 추출된 top_n index들 출력. top_n index는 2차원 데이터 임. 
    #dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]
```


```python
similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)

similar_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>genres</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>popularity</th>
      <th>keywords</th>
      <th>overview</th>
      <th>genres_literal</th>
      <th>weighted_vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>240</td>
      <td>The Godfather: Part II</td>
      <td>[Drama, Crime]</td>
      <td>8.3</td>
      <td>3338</td>
      <td>105.792936</td>
      <td>[italo-american, cuba, vororte, melancholy, praise, revenge, mafia, lawyer, blood, corrupt polit...</td>
      <td>In the continuing saga of the Corleone crime family, a young Vito Corleone grows up in Sicily an...</td>
      <td>Drama Crime</td>
      <td>8.079586</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>769</td>
      <td>GoodFellas</td>
      <td>[Drama, Crime]</td>
      <td>8.2</td>
      <td>3128</td>
      <td>63.654244</td>
      <td>[prison, based on novel, florida, 1970s, mass murder, irish-american, drug traffic, biography, b...</td>
      <td>The true story of Henry Hill, a half-Irish, half-Sicilian Brooklyn kid who is adopted by neighbo...</td>
      <td>Drama Crime</td>
      <td>7.976937</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>598</td>
      <td>City of God</td>
      <td>[Drama, Crime]</td>
      <td>8.1</td>
      <td>1814</td>
      <td>44.356711</td>
      <td>[male nudity, street gang, brazilian, photographer, 1970s, puberty, ghetto, gang war, coming of ...</td>
      <td>Cidade de Deus is a shantytown that started during the 1960s and became one of Rio de Janeiro’s ...</td>
      <td>Drama Crime</td>
      <td>7.759693</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>311</td>
      <td>Once Upon a Time in America</td>
      <td>[Drama, Crime]</td>
      <td>8.2</td>
      <td>1069</td>
      <td>49.336397</td>
      <td>[life and death, corruption, street gang, rape, sadistic, lovesickness, sexual abuse, money laun...</td>
      <td>A former Prohibition-era Jewish gangster returns to the Lower East Side of Manhattan over thirty...</td>
      <td>Drama Crime</td>
      <td>7.657811</td>
    </tr>
    <tr>
      <th>883</th>
      <td>640</td>
      <td>Catch Me If You Can</td>
      <td>[Drama, Crime]</td>
      <td>7.7</td>
      <td>3795</td>
      <td>73.944049</td>
      <td>[con man, biography, fbi agent, overhead camera shot, attempted jailbreak, engagement party, mis...</td>
      <td>A true story about Frank Abagnale Jr. who, before his 19th birthday, successfully conned million...</td>
      <td>Drama Crime</td>
      <td>7.557097</td>
    </tr>
    <tr>
      <th>281</th>
      <td>4982</td>
      <td>American Gangster</td>
      <td>[Drama, Crime]</td>
      <td>7.4</td>
      <td>1502</td>
      <td>42.361215</td>
      <td>[underdog, black people, drug traffic, drug smuggle, society, ambition, rise and fall, cop, drug...</td>
      <td>Following the death of his employer and mentor, Bumpy Johnson, Frank Lucas establishes himself a...</td>
      <td>Drama Crime</td>
      <td>7.141396</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>11798</td>
      <td>This Is England</td>
      <td>[Drama, Crime]</td>
      <td>7.4</td>
      <td>363</td>
      <td>8.395624</td>
      <td>[holiday, skinhead, england, vandalism, independent film, gang, racism, summer, youth, violence,...</td>
      <td>A story about a troubled boy growing up in England, set in 1983. He comes across a few skinheads...</td>
      <td>Drama Crime</td>
      <td>6.739664</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>168672</td>
      <td>American Hustle</td>
      <td>[Drama, Crime]</td>
      <td>6.8</td>
      <td>2807</td>
      <td>49.664128</td>
      <td>[con artist, scam, mobster, fbi agent]</td>
      <td>A con man, Irving Rosenfeld, along with his seductive partner Sydney Prosser, is forced to work ...</td>
      <td>Drama Crime</td>
      <td>6.717525</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>203</td>
      <td>Mean Streets</td>
      <td>[Drama, Crime]</td>
      <td>7.2</td>
      <td>345</td>
      <td>17.002096</td>
      <td>[epilepsy, protection money, secret love, money, redemption]</td>
      <td>A small-time hood must choose from among love, friendship and the chance to rise within the mob.</td>
      <td>Drama Crime</td>
      <td>6.626569</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>10220</td>
      <td>Rounders</td>
      <td>[Drama, Crime]</td>
      <td>6.9</td>
      <td>439</td>
      <td>18.422008</td>
      <td>[gambling, law, compulsive gambling, roulette, gain]</td>
      <td>A young man is a reformed gambler who must return to playing big stakes poker to help a friend p...</td>
      <td>Drama Crime</td>
      <td>6.530427</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
# 유사도를 나타내는 특징 컬럼을 데이터프레임에 추가

movies_df[['title','vote_average','vote_count']].sort_values('vote_average', ascending=False)[:10]
C = movies_df['vote_average'].mean()
m = movies_df['vote_count'].quantile(0.6)
percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
C = movies_df['vote_average'].mean()
```

<br>

### 3. 최종 유사도 계산


```python
def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )
```


```python
# 유사도 가중치를 나타내는 특징 컬럼을 데이터프레임에 추가

movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1) 
movies_df[['title','vote_average','weighted_vote','vote_count']].sort_values('weighted_vote',
                                                                          ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
      <th>weighted_vote</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8.5</td>
      <td>8.396052</td>
      <td>8205</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>8.4</td>
      <td>8.263591</td>
      <td>5893</td>
    </tr>
    <tr>
      <th>662</th>
      <td>Fight Club</td>
      <td>8.3</td>
      <td>8.216455</td>
      <td>9413</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Pulp Fiction</td>
      <td>8.3</td>
      <td>8.207102</td>
      <td>8428</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Dark Knight</td>
      <td>8.2</td>
      <td>8.136930</td>
      <td>12002</td>
    </tr>
    <tr>
      <th>1818</th>
      <td>Schindler's List</td>
      <td>8.3</td>
      <td>8.126069</td>
      <td>4329</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>Whiplash</td>
      <td>8.3</td>
      <td>8.123248</td>
      <td>4254</td>
    </tr>
    <tr>
      <th>809</th>
      <td>Forrest Gump</td>
      <td>8.2</td>
      <td>8.105954</td>
      <td>7927</td>
    </tr>
    <tr>
      <th>2294</th>
      <td>Spirited Away</td>
      <td>8.3</td>
      <td>8.105867</td>
      <td>3840</td>
    </tr>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
      <td>8.079586</td>
      <td>3338</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### 4. 장르 유사성이 높은 영화 추천


```python
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]
```


```python
similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title', 'vote_average', 'weighted_vote']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
    vertical-align: top;
}





</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
      <th>weighted_vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
      <td>8.079586</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
      <td>7.976937</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
      <td>7.759693</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>Once Upon a Time in America</td>
      <td>8.2</td>
      <td>7.657811</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
      <td>7.557097</td>
    </tr>
    <tr>
      <th>281</th>
      <td>American Gangster</td>
      <td>7.4</td>
      <td>7.141396</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>This Is England</td>
      <td>7.4</td>
      <td>6.739664</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>American Hustle</td>
      <td>6.8</td>
      <td>6.717525</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
      <td>6.626569</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>Rounders</td>
      <td>6.9</td>
      <td>6.530427</td>
    </tr>
  </tbody>
</table>
</div>

