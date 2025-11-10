def build_phoc(word: str):
    """
    Build a PHOC (Pyramidal Histogram of Characters) vector for a UTF-8 string.
    Supports Vietnamese letters, digits, and punctuation (mapped to a single PUNCT bucket).
    Returns a list of floats (0.0 or 1.0) of fixed dimension 604.
    """
    # --- Definitions ---
    # Base unigrams: Vietnamese letters and digits (35 entries)
    base_unigrams = [
        'a','ă','â','b','c','d','đ','e','ê','f','g','h','i','j','k',
        'l','m','n','o','ô','ơ','p','q','r','s','t','u','ư','v','w','x','y','z',
        'á','ắ','ấ','é','ế','í','ó','ố','ớ','ú','ứ','ý',
        'à','ằ','ầ','è','ề','ì','ò','ồ','ờ','ù','ừ','ỳ',
        'ả','ẳ','ẩ','ẻ','ể','ỉ','ỏ','ổ','ở','ủ','ử','ỷ',
        'ã','ẵ','ẫ','ẽ','ễ','ĩ','õ','ỗ','ỡ','ũ','ữ','ỹ',
        'ạ','ặ','ậ','ẹ','ệ','ị','ọ','ộ','ợ','ụ','ự','ỵ',
        '0','1','2','3','4','5','6','7','8','9',
        '.', ',', ';', ':', '?', '!', '"', "'", '-', '(', ')', '[', ']', '{', '}', '/', '\\',
        ' '
    ]
    # Add one PUNCT bucket for all punctuation
    unigrams = base_unigrams + ['PUNCT']  # total 36

    # Punctuation set
    punctuation_set = set(['.', ',', ';', ':', '?', '!', '"', "'", '-', '(', ')', '[', ']', '{', '}', '/', '\\'])

    # Bigrams (50 entries)
    bigrams = [
        'ng','th','ch','nh','tr','qu','gi','ph','kh','gh',
        'ai','ao','au','ay','eo','eu','ia','ie','iu',
        'oa','oe','oi','ua','ue','ui',
        'an','em','in','on','oc','uc','at','en','es','el',
        'al','ol','ul','il','im','um','om','up','ap','êt',
        'ăm','ắn','ơn','ươ','ưa','uâ'
    ]

    # Constants
    levels = [2, 3, 4, 5]  # unigram pyramid levels
    N_uni = len(unigrams)  # 36
    N_bi = len(bigrams)    # 50
    total_regions = sum(levels)  # 14

    # PHOC vector length: 14*36 + 2*50 = 504 + 100 = 604
    phoc_len = total_regions * N_uni + 2 * N_bi
    phoc = [0.0] * phoc_len

    n = len(word)
    if n == 0:
        return phoc

    # --- Unigram Encoding ---
    for idx, ch in enumerate(word):
        # Determine unigram index
        if ch in punctuation_set:
            cidx = unigrams.index('PUNCT')
        else:
            try:
                cidx = unigrams.index(ch)
            except ValueError:
                # Skip unknown characters
                continue

        occ0 = idx / n
        occ1 = (idx + 1) / n

        region_offset = 0
        for lvl in levels:
            for rg in range(lvl):
                r0 = rg / lvl
                r1 = (rg + 1) / lvl
                overlap = min(occ1, r1) - max(occ0, r0)
                if overlap / (occ1 - occ0) >= 0.5:
                    feat_idx = (region_offset + rg) * N_uni + cidx
                    phoc[feat_idx] = 1.0
            region_offset += lvl

    # --- Bigram Encoding (level 2) ---
    bigram_offset = total_regions * N_uni
    for i in range(n - 1):
        bg = word[i:i+2]
        try:
            bidx = bigrams.index(bg)
        except ValueError:
            continue

        occ0 = i / n
        occ1 = (i + 2) / n
        for rg in range(2):  # only level 2
            r0 = rg / 2
            r1 = (rg + 1) / 2
            overlap = min(occ1, r1) - max(occ0, r0)
            if overlap / (occ1 - occ0) >= 0.5:
                feat_idx = bigram_offset + rg * N_bi + bidx
                phoc[feat_idx] = 1.0

    return phoc
