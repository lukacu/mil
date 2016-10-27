// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.


#ifndef H_MATRIX
#define H_MATRIX

#include "Public.h"

template<class T> class Matrix;
typedef Matrix<float>	Matrixf;
typedef Matrix<uchar>	Matrixu;

template<class T> class Matrix
{

public:
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // members
    int				_rows, _cols, _depth;
    vector<void*>	_data;

private:
    // image specific

    int				_dataSize;

    Mat		_iplimg;
    vector<float*>	_iidata;
    int _iidataSize;
    int				_iipixStep;
    bool			_ii_init;

    Size		_roi; //whole image roi (needed for some functions)
    Rect		_roirect;

public:
    bool			_keepIpl;  // if set to true, calling freeIpl() will have no effect;  this is for speed up only...

public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, int depth);
    Matrix(const Matrix<T>& x);
    Matrix(const vector<T>& v);
    ~Matrix();
    static		Matrix<T>	Eye(int sz);
    void		Resize(uint rows, uint cols, uint depth = 1);
    void		Resize(uint depth);
    void		Free();
    void		Set(T val);
    void		Set(T val, int channel);

    T&			operator()(const int k) const;
    T&			operator()(const int row, const int col) const;
    T&			operator()(const int row, const int col, const int depth) const;
    vector<T>	operator()(const vectori rows, const vectori cols);
    vector<T>	operator()(const vectori rows, const vectori cols, const vectori depths);
    float		ii(const int row, const int col, const int depth) const;
    Matrix<T>	getCh(uint ch);
    Mat	getIpl()
    {
        return _iplimg;
    };

    int			rows() const
    {
        return _rows;
    };
    int			cols() const
    {
        return _cols;
    };
    int			depth() const
    {
        return _depth;
    };
    uint		size() const
    {
        return _cols * _rows;
    };
    int			length() const
    {
        return MAX(_cols, _rows);
    };

    Matrix<T>&	operator= (const Matrix<T>& x);
    Matrix<T>&	operator= (const vector<T>& x);
    Matrix<T>	operator+ (const Matrix<T>& b) const;
    Matrix<T>	operator+ (const T& a) const;
    Matrix<T>	operator- (const Matrix<T>& b) const;
    Matrix<T>	operator- (const T& a) const;
    Matrix<T>	operator* (const T& a) const;
    Matrix<T>	operator& (const Matrix<T>& b) const;
    Matrixu		operator< (const T& a) const;
    Matrixu		operator> (const T& a) const;
    Matrix<T>	normalize() const;
    Matrix<T>	Sqr() const;
    Matrix<T>	Exp() const;
    void		Trans(Matrix<T>& res);
    T			Max(uint channel = 0) const;
    T			Min(uint channel = 0) const;
    double		Sum(uint channel = 0) const;
    void		Max(T& val, uint& row, uint& col, uint channel = 0) const;
    void		Min(T& val, uint& row, uint& col, uint channel = 0) const;
    float		Mean(uint channel = 0) const;
    float		Var(uint channel = 0) const;
    float		VarW(const Matrixf& w, T *mu = NULL) const;
    float		MeanW(const vectorf& w)  const;
    float		MeanW(const Matrixf& w)  const;
    float		Dot(const Matrixf& x);

    void		initII();
    bool		isInitII() const
    {
        return _ii_init;
    };
    void		FreeII();
    float		sumRect(const Rect& rect, int channel) const;
    void		drawRect(Rect rect, int lineWidth = 3, int R = 255, int G = 0, int B = 0);
    void		drawRect(float width, float height, float x, float y, float sc, float th, int lineWidth = 3, int R = 255, int G = 0, int B = 0);
    void		drawEllipse(float height, float width, float x, float y, int lineWidth = 3, int R = 255, int G = 0, int B = 0);
    void		drawEllipse(float height, float width, float x, float y, float startang, float endang, int lineWidth = 3, int R = 255, int G = 0, int B = 0);
    void		drawText(const char* txt, float x, float y, int R = 255, int G = 255, int B = 0);
    void		warp(Matrixu& res, uint rows, uint cols, float x, float y, float sc = 1.0f, float th = 0.0f, float sr = 1.0f, float phi = 0.0f);
    void		warpAll(uint rows, uint cols, vector<vectorf> params, vector<Matrixu>& res);
    void		computeGradChannels();
    Matrixu		imResize(float p, float x = -1);
    void		conv2RGB(Matrixu& res);
    void		conv2BW(Matrixu& res);
    float		dii_dx(uint x, uint y, uint channel = 0);
    float		dii_dy(uint x, uint y, uint channel = 0);

    void		createIpl(bool force = false);
    void		freeIpl();


    static Matrix<T>		vecMat2Mat(const vector<Matrix<T> >& x);
    static vector<Matrix<T> >	vecMatTranspose(const vector<Matrix<T> >& x);

    Matrixu		convert2img(float min = 0.0f, float max = 0.0f);


    void		IplImage2Matrix(Mat& img);
    void		GrayIplImage2Matrix(Mat& img);


};

//template<> void					Matrixu::createIpl(bool force);

template<class T> ostream&			operator<< (ostream& os, const Matrix<T>& x);


template<class T>					Matrix<T>::Matrix()
{
    _rows		= 0;
    _cols		= 0;
    _depth		= 0;
    _keepIpl	= false;
}

template<class T>					Matrix<T>::Matrix(int rows, int cols)
{
    _rows		= 0;
    _cols		= 0;
    _depth		= 0;
    _keepIpl	= false;
    _ii_init	= false;
    Resize(rows, cols, 1);
}

template<class T>					Matrix<T>::Matrix(int rows, int cols, int depth)
{
    _rows		= 0;
    _cols		= 0;
    _depth		= 0;
    _keepIpl	= false;
    _ii_init	= false;
    Resize(rows, cols, depth);
}

template<class T>					Matrix<T>::Matrix(const Matrix<T>& a)
{
    _rows		= 0;
    _cols		= 0;
    _depth		= 0;
    //_iplimg		= NULL;
    _keepIpl	= (typeid(T) == typeid(uchar)) && a._keepIpl;
    _ii_init	= false;
    Resize(a._rows, a._cols, a._depth);
    if(typeid(T) == typeid(uchar))
    {
        for(uint k = 0; k < _data.size(); k++)
            memcpy(_data[k], a._data[k], _dataSize);
    }
    else
    {
        for(uint k = 0; k < _data.size(); k++)
            memcpy(_data[k], a._data[k], _dataSize);
    }
    if(a._ii_init)
    {
        _iidata.resize(a._iidata.size());
        _iidataSize = (_rows + 1) * (_cols + 1) * sizeof(float);
        for(uint k = 0; k < _iidata.size(); k++)
        {
            if(_iidata[k] != NULL) free(_iidata[k]);
            _iidata[k] = (float *)malloc(_iidataSize);
            _iipixStep = _iidataSize / sizeof(float);
            memcpy(_iidata[k], a._iidata[k], _iidataSize);
        }
        _ii_init = true;
    }

    if(!a._iplimg.empty() && typeid(T) == typeid(uchar))
    {
        ((Matrixu*)this)->createIpl();
        a._iplimg.assignTo(_iplimg);
    }
}

template<class T> Matrix<T>	Matrix<T>::Eye(int sz)
{
    Matrix<T> res(sz, sz);
    for(int k = 0; k < sz; k++)
        res(k, k) = 1;
    return res;
}

template<class T> void Matrix<T>::Resize(uint rows, uint cols, uint depth)
{
    if(rows < 0 || cols < 0)
        abortError(__LINE__, __FILE__, "NEGATIVE MATRIX SIZE");

    if(_rows == rows && _cols == cols && _depth == depth) return;
    bool err = false;
    Free();
    _rows = rows;
    _cols = cols;
    _depth = depth;

    _data.resize(depth);

    if(typeid(T) == typeid(uchar))
    {
        _dataSize = _rows * _cols * sizeof(uchar);
    }
    else
    {
        _dataSize = _rows * _cols * sizeof(float);
    }

    for(uint k = 0; k < _data.size(); k++)
    {
        _data[k] = malloc((uint)_dataSize);
        err = err || _data[k] == NULL;
    }

    _roi.width = cols;
    _roi.height = rows;
    _roirect.width = cols;
    _roirect.height = rows;
    _roirect.x = 0;
    _roirect.y = 0;
    Set(0);

    //free ipl
    if(!_iplimg.empty())
        _iplimg.release();

    if(err)
        abortError(__LINE__, __FILE__, "OUT OF MEMORY");
}

template<class T> void Matrix<T>::Resize(uint depth)
{

    if(_depth == depth) return;
    bool err = false;


    _data.resize(depth);

    for(uint k = _depth; k < depth; k++)
    {
        _data[k] = malloc((uint)_dataSize);
        err = err || _data[k] == NULL;
        Set(0, k);
    }
    _depth = depth;


    if(err)
        abortError(__LINE__, __FILE__, "OUT OF MEMORY");
}

template<class T> void Matrix<T>::Free()
{
    if(_ii_init) FreeII();
    if(!_iplimg.empty()) _iplimg.release();
    _ii_init = false;

    for(uint k = 0;  k < _data.size(); k++)
        if(_data[k] != NULL)
            if(typeid(T) == typeid(uchar))
            {
                free(_data[k]);
            }
            else
            {
                free(_data[k]);
            }

    _rows = 0;
    _cols = 0;
    _depth = 0;

    _data.resize(0);
}

template<class T> void Matrix<T>::Set(T val)
{


    for(uint k = 0; k < _data.size(); k++)
        if(typeid(T) == typeid(uchar))
        {
            memset(_data[k], val, _dataSize);
        }
        else
        {
            for(uint j = 0; j < (uint)_rows * _cols; j++)
                ((float*)_data[k])[j] = val;
        }

}
template<class T> void Matrix<T>::Set(T val, int k)
{

    if(typeid(T) == typeid(uchar))
    {
        memset(_data[k], val, _dataSize);
    }
    else
    {
        for(uint j = 0; j < (uint)_rows * _cols; j++)
            ((float*)_data[k])[j] = val;
    }
}


template<class T> void				Matrix<T>::FreeII()
{
    for(uint k = 0;  k < _iidata.size(); k++)
        free(_iidata[k]);
    _iidata.resize(0);
    _ii_init = false;
}

template<class T>					Matrix<T>::~Matrix()
{
    Free();
}

template<class T> inline T&			Matrix<T>::operator()(const int row, const int col, const int depth) const
{

    if(typeid(T) == typeid(uchar))
        return (T&)((uchar*)_data[depth])[row * (_cols) + col];
    else
        return (T&)((float*)_data[depth])[row * (_cols) + col];

}

template<class T> inline T&			Matrix<T>::operator()(const int row, const int col) const
{
    return (*this)(row, col, 0);
}
template<class T> inline T&			Matrix<T>::operator()(const int k) const
{
    return (*this)(k / _cols, k % _cols, 0);
}
template<class T> inline vector<T>	Matrix<T>::operator()(const vectori rows, const vectori cols)
{
    assert(rows.size() == cols.size());
    vector<T> res;
    res.resize(rows.size());
    for(uint k = 0; k < rows.size(); k++)
        res[k] = (*this)(rows[k], cols[k]);
    return res;
}

template<class T> inline vector<T>	Matrix<T>::operator()(const vectori rows, const vectori cols, const vectori depths)
{
    assert(rows.size() == cols.size() && cols.size() == depths.size());
    vector<T> res;
    res.resize(rows.size());
    for(uint k = 0; k < rows.size(); k++)
        res[k] = (*this)(rows[k], cols[k], depths[k]);
    return res;
}

template<class T> inline Matrix<T>	Matrix<T>::getCh(uint ch)
{
    Matrix<T> a(_rows, _cols, 1);
    memcpy(a._data[0], _data[ch], _dataSize);

    return a;
}
template<class T> Matrix<T>&		Matrix<T>::operator= (const Matrix<T>& a)
{
    if(this != &a)
    {
        Resize(a._rows, a._cols, a._depth);
        for(uint k = 0; k < _data.size(); k++)
            memcpy(_data[k], a._data[k], _dataSize);

        //ippmCopy_va_32f_SS((float*)a._data[k],sizeof(float)*_cols,sizeof(float),(float*)_data[k],sizeof(float)*_cols,sizeof(float),_cols,_rows);
        if(a._ii_init)
        {
            _iidata.resize(a._iidata.size());
            _iidataSize = (_rows + 1) * (_cols + 1) * sizeof(float);
            for(uint k = 0; k < _iidata.size(); k++)
            {
                if(_iidata[k] != NULL) free(_iidata[k]);
                _iidata[k] = (float *) malloc(_iidataSize);
                _iipixStep = _iidataSize / sizeof(float);
                memcpy(_iidata[k], a._iidata[k], _iidataSize);
            }
            _ii_init = true;
        }

    }
    return (*this);
}





template<class T> Matrix<T>&		Matrix<T>::operator= (const vector<T>& a)
{
    Resize(1, a.size(), 1);
    for(uint k = 0; k < a.size(); k++)
        (*this)(k) = a[k];

    return (*this);
}





template<class T> Matrix<T>			Matrix<T>::operator+ (const Matrix<T>& a) const
{
    Matrix<T> res(rows(), cols());
    assert(rows() == a.rows() && cols() == a.cols());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m2(_rows, _cols, type, a._data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        m3 = m1 + m2;
    }

    return res;
}

template<class T> Matrix<T>			Matrix<T>::operator+ (const T& a) const
{
    Matrix<T> res;
    res.Resize(rows(), cols(), depth());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        m3 = m1 + a;
    }
    return res;
}
template<class T> Matrix<T>			Matrix<T>::operator- (const Matrix<T>& a) const
{
    return (*this) + (a * -1);
}

template<class T> Matrix<T>			Matrix<T>::operator- (const T& a) const
{
    return (*this) + (a * -1);
}
template<class T> Matrix<T>			Matrix<T>::operator* (const T& a) const
{
    Matrix<T> res(rows(), cols(), depth());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        m3 = m1 * a;
    }
    return res;
}
template<class T> Matrix<T>			Matrix<T>::operator& (const Matrix<T>& b) const
{
    Matrix<T> res(rows(), cols(), depth());
    assert(rows() == b.rows() && cols() == b.cols() && depth() == b.depth());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m2(_rows, _cols, type, b._data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        m3 = m1.mul(m2);
    }
    return res;
}

template<class T> Matrixu			Matrix<T>::operator< (const T& b) const
{
    Matrixu res(rows(), cols());

    for(uint i = 0; i < size(); i++)
        res(i) = (uint)((*this)(i) < b);

    return res;
}

template<class T> Matrixu			Matrix<T>::operator> (const T& b) const
{
    Matrixu res(rows(), cols());

    for(uint i = 0; i < size(); i++)
        res(i) = (uint)((*this)(i) > b);

    return res;
}
template<class T> Matrix<T>			Matrix<T>::normalize() const
{
    double sum = this->Sum();
    return (*this) * (T)(1.0 / (sum + 1e-6));
}
template<class T> Matrix<T>			Matrix<T>::Sqr() const
{
    Matrix<T> res(rows(), cols(), depth());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        pow(m1, 2, m3);
    }
    return res;
}
template<class T> Matrix<T>			Matrix<T>::Exp() const
{
    Matrix<T> res(rows(), cols(), depth());

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        exp(m1, m3);
    }

    return res;
}
template<class T> ostream&			operator<<(ostream& os, const Matrix<T>& x)
{
    //display matrix
    os << "[ ";
    char tmp[1024];
    for(int j = 0; j < x.rows(); j++)
    {
        if(j > 0) os << "  ";
        for(int i = 0; i < x.cols(); i++)
        {
            if(typeid(T) == typeid(uchar))
                sprintf(tmp, "%3d", (int)x(j, i));
            else
                sprintf(tmp, "%02.2f", (float)x(j, i));
            os << tmp << " ";
        }
        if(j != x.rows() - 1)
            os << "\n";
    }
    os << "]";
    return os;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T> void				Matrix<T>::Trans(Matrix<T>& res)
{
    res.Resize(_cols, _rows, _depth);
    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    for(uint k = 0; k < _data.size(); k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(_rows, _cols, type, res._data[k]);

        transpose(m1, m3);
    }

}

template<class T> T					Matrix<T>::Max(uint channel) const
{
    double maxVal;

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;
    Mat m1(_rows, _cols, type, _data[channel]);

    minMaxLoc(m1, NULL, &maxVal);

    return maxVal;
}

template<class T> T					Matrix<T>::Min(uint channel)  const
{
    double minVal;

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;
    Mat m1(_rows, _cols, type, _data[channel]);

    minMaxLoc(m1, &minVal);

    return minVal;
}




template<class T> void				Matrix<T>::Max(T& val, uint& row, uint& col, uint channel) const
{
    Point pos;

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;
    Mat m1(_rows, _cols, type, _data[channel]);

    minMaxLoc(m1, NULL, &val, NULL, &pos);

    row = pos.y;
    col = pos.x;

}

template<class T> void				Matrix<T>::Min(T& val, uint& row, uint& col, uint channel)  const
{
    Point pos;

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;
    Mat m1(_rows, _cols, type, _data[channel]);

    minMaxLoc(m1, &val, NULL, &pos, NULL);

    row = pos.y;
    col = pos.x;
}


template<class T> float				Matrix<T>::Mean(uint channel)  const
{

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    Mat m1(_rows, _cols, type, _data[channel]);

    return mean(m1)[0];
}

template<class T> float				Matrix<T>::Var(uint channel)  const
{
    Scalar mean, stddev;

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    Mat m1(_rows, _cols, type, _data[channel]);

    meanStdDev(m1, mean, stddev);

    return (float)(stddev[0] * stddev[0]);
}

template<class T> float				Matrix<T>::VarW(const Matrixf& w, T *mu)  const
{
    T mm;
    if(mu == NULL)
        mm = (*this).MeanW(w);
    else
        mm = *mu;
    return ((*this) - mm).Sqr().MeanW(w);
}

template<class T> float				Matrix<T>::MeanW(const vectorf& w)  const
{
    float mean = 0.0f;
    assert(w.size() == this->size());
    for(uint k = 0; k < w.size(); k++)
        mean += w[k] * (*this)(k);

    return mean;
}

template<class T> float				Matrix<T>::MeanW(const Matrixf& w)  const
{
    return (float)((*this)&w).Sum();
}
template<class T> double			Matrix<T>::Sum(uint channel)  const
{

    int type = (typeid(T) == typeid(uchar)) ? CV_8UC1 : CV_32FC1;

    Mat m1(_rows, _cols, type, _data[channel]);

    return sum(m1)[0];
}


template<class T> Matrixu			Matrix<T>::convert2img(float min, float max)
{
    if(max == min)
    {
        max = Max();
        min = Min();
    }

    Matrixu res(rows(), cols());
    // scale to 0 to 255
    Matrix<T> tmp;
    tmp = (*this);
    tmp = (tmp - (T)min) * (255 / ((T)max - (T)min));

    //#pragma omp parallel for
    for(int d = 0; d < depth(); d++)
        for(int row = 0; row < rows(); row++)
            for(int col = 0; col < cols(); col++)
                res(row, col) = (uchar)tmp(row, col);

    return res;

}

template<class T> Matrix<T>			Matrix<T>::vecMat2Mat(const vector<Matrix<T> >& x)
{
    Matrix<T> t(x.size(), x[0].size());

    #pragma omp parallel for
    for(int k = 0; k < (int)t.rows(); k++)
        for(int j = 0; j < t.cols(); j++)
            t(k, j) = x[k](j);

    return t;
}
template<class T> vector<Matrix<T> >	Matrix<T>::vecMatTranspose(const vector<Matrix<T> >& x)
{
    vector<Matrix<T> > t(x[0].size());

    #pragma omp parallel for
    for(int k = 0; k < (int)t.size(); k++)
        t[k].Resize(1, x.size());

    #pragma omp parallel for
    for(int k = 0; k < (int)t.size(); k++)
        for(uint j = 0; j < x.size(); j++)
            t[k](j) = x[j](k);


    return t;
}

#endif

