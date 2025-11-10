#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

// Get length of next UTF-8 character
int utf8_charlen(const char *s) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) return 1;
    else if ((c >> 5) == 0x6) return 2;
    else if ((c >> 4) == 0xE) return 3;
    else if ((c >> 3) == 0x1E) return 4;
    return 1;  // fallback
}

// Compare UTF-8 strings
int utf8_cmp(const char *a, const char *b) {
    int la = utf8_charlen(a);
    int lb = utf8_charlen(b);
    if (la != lb) return -1;
    return memcmp(a, b, la);
}

static PyObject *build_phoc(PyObject *self, PyObject *args) {
    const char *word = NULL;
    if (!PyArg_ParseTuple(args, "s", &word)) {
        return PyErr_Format(PyExc_RuntimeError, "Invalid argument. Must be a string.");
    }

    // Define unigrams (UTF-8)
    const char *unigrams[] = {
        "a","ă","â","b","c","d","đ","e","ê","f","g","h","i","j","k",
        "l","m","n","o","ô","ơ","p","q","r","s","t","u","ư","v","w","x","y","z",
        "á","ắ","ấ","é","ế","í","ó","ố","ớ","ú","ứ","ý",
        "à","ằ","ầ","è","ề","ì","ò","ồ","ờ","ù","ừ","ỳ",
        "ả","ẳ","ẩ","ẻ","ể","ỉ","ỏ","ổ","ở","ủ","ử","ỷ",
        "ã","ẵ","ẫ","ẽ","ễ","ĩ","õ","ỗ","ỡ","ũ","ữ","ỹ",
        "ạ","ặ","ậ","ẹ","ệ","ị","ọ","ộ","ợ","ụ","ự","ỵ",
        "0","1","2","3","4","5","6","7","8","9"
    };
    int N_UNIGRAMS = sizeof(unigrams) / sizeof(unigrams[0]);

    const char *bigrams[] = {
        "ng", "th", "ch", "nh", "tr", "qu", "gi", "ph", "kh", "gh",
        "ai", "ao", "au", "ay", "eo", "eu", "ia", "ie", "iu",
        "oa", "oe", "oi", "ua", "ue", "ui",
        "an", "em", "in", "on", "oc", "uc", "at", "en", "es", "el",
        "al", "ol", "ul", "il", "im", "um", "om", "up", "ap", "êt",
        "ăm", "ắn", "ơn", "ươ", "ưa", "uâ"
    };
    int N_BIGRAMS = sizeof(bigrams) / sizeof(bigrams[0]);

    float phoc[604] = {.0};

    // Parse UTF-8 word into character array
    const char *p = word;
    const char *utf_chars[128];
    int char_pos[128];  // Byte position of each char
    int n = 0;

    while (*p) {
        utf_chars[n] = p;
        char_pos[n] = p - word;
        int len = utf8_charlen(p);
        p += len;
        n++;
    }

    for (int index = 0; index < n; index++) {
        float char_occ0 = (float)index / n;
        float char_occ1 = (float)(index + 1) / n;

        int char_index = -1;
        for (int k = 0; k < N_UNIGRAMS; k++) {
            if (utf8_cmp(unigrams[k], utf_chars[index]) == 0) {
                char_index = k;
                break;
            }
        }
        if (char_index == -1) {
            return PyErr_Format(
                PyExc_RuntimeError,
                "Unknown character: '%.*s'", utf8_charlen(utf_chars[index]), utf_chars[index]);
        }

        for (int level = 2; level < 6; level++) {
            for (int region = 0; region < level; region++) {
                float region_occ0 = (float)region / level;
                float region_occ1 = (float)(region + 1) / level;
                float overlap0 = max(char_occ0, region_occ0);
                float overlap1 = min(char_occ1, region_occ1);
                float ratio = (overlap1 - overlap0) / (char_occ1 - char_occ0);
                if (ratio >= 0.5) {
                    int sum = 0;
                    for (int l = 2; l < level; l++) sum += l;
                    int feat_vec_index = sum * 36 + region * 36 + char_index;
                    phoc[feat_vec_index] = 1;
                }
            }
        }
    }

    // Bigrams
    int ngram_offset = 36 * 14;
    for (int i = 0; i < n - 1; i++) {
        char bigram_buf[8] = {0};
        int l1 = utf8_charlen(utf_chars[i]);
        int l2 = utf8_charlen(utf_chars[i + 1]);
        memcpy(bigram_buf, utf_chars[i], l1);
        memcpy(bigram_buf + l1, utf_chars[i + 1], l2);

        int ngram_index = -1;
        for (int k = 0; k < N_BIGRAMS; k++) {
            if (memcmp(bigrams[k], bigram_buf, l1 + l2) == 0) {
                ngram_index = k;
                break;
            }
        }

        if (ngram_index == -1) continue;

        float ngram_occ0 = (float)i / n;
        float ngram_occ1 = (float)(i + 2) / n;

        for (int region = 0; region < 2; region++) {
            float region_occ0 = (float)region / 2;
            float region_occ1 = (float)(region + 1) / 2;
            float overlap0 = max(ngram_occ0, region_occ0);
            float overlap1 = min(ngram_occ1, region_occ1);
            if ((overlap1 - overlap0) / (ngram_occ1 - ngram_occ0) >= 0.5) {
                phoc[ngram_offset + region * N_BIGRAMS + ngram_index] = 1;
            }
        }
    }

    PyObject *dlist = PyList_New(604);
    for (int i = 0; i < 604; i++) {
        PyList_SetItem(dlist, i, PyFloat_FromDouble(phoc[i]));
    }

    return dlist;
}

static PyObject *getList(PyObject *self, PyObject *args)
{
    PyObject *dlist = PyList_New(2);
    PyList_SetItem(dlist, 0, PyFloat_FromDouble(0.00001));
    PyList_SetItem(dlist, 1, PyFloat_FromDouble(42.0));

    return dlist;
}

static PyMethodDef myMethods[] = {
    {"build_phoc", build_phoc, METH_VARARGS, ""},
    {"getList", getList, METH_NOARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cphoc_vn_module = {
    PyModuleDef_HEAD_INIT,
    "cphoc_vn",          // <-- match this to your .so filename
    "cphoc_vn Module",   // docstring
    -1,
    myMethods
};

// Rename this to match “cphoc_vn”
PyMODINIT_FUNC
PyInit_cphoc_vn(void)  // <-- was PyInit_cphoc before
{
    return PyModule_Create(&cphoc_vn_module);
}