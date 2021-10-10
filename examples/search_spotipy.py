def main():
    from spotipy.oauth2 import SpotifyClientCredentials
    import spotipy
    import sys
    import pprint



    cid = '223f187f808f45ecb62cbf9534e81c77'
    secret = 'b178324b16ff4bb3bda66798ec6ece37'
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret))

    search_str = 'happy'
    result = sp.search(search_str)
    pprint.pprint(result)

if __name__=='__main__':
    main()