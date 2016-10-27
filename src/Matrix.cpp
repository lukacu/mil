// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Matrix.h"

template<> void					Matrixu::createIpl(bool force)
{

    _iplimg.create(Size(_cols, _rows), CV_8UC3);

    if(_depth == 1)
        for(int row = 0; row < _rows; row++)
            for(int k = 0; k < _cols * 3; k += 3)
            {
                _iplimg.data[row * _iplimg.step + k + 2] = ((uchar*)_data[0])[row * _cols + k / 3];
                _iplimg.data[row * _iplimg.step + k + 1] = ((uchar*)_data[0])[row * _cols + k / 3];
                _iplimg.data[row * _iplimg.step + k  ] = ((uchar*)_data[0])[row * _cols + k / 3];
            }
    else
        for(int row = 0; row < _rows; row++)
            for(int k = 0; k < _cols * 3; k += 3)
            {
                _iplimg.data[row * _iplimg.step + k + 2] = ((uchar*)_data[0])[row * _cols + k / 3];
                _iplimg.data[row * _iplimg.step + k + 1] = ((uchar*)_data[1])[row * _cols + k / 3];
                _iplimg.data[row * _iplimg.step + k  ] = ((uchar*)_data[2])[row * _cols + k / 3];
            }

}

template<> void					Matrixu::freeIpl()
{
    if(!_keepIpl) _iplimg.release();
}

template<> void					Matrixu::IplImage2Matrix(Mat& img)
{
    //Resize(img->height, img->width, img->nChannels);
    bool origin = false; //img.origin==1;

    if(_depth == 1)
        for(int row = 0; row < _rows; row++)
            for(int k = 0; k < _cols * 3; k += 3)
                if(origin)
                    ((uchar*)_data[0])[(_rows - row - 1) * _cols + k / 3] = img.data[row * img.step + k];
                else
                    ((uchar*)_data[0])[row * _cols + k / 3] = img.data[row * img.step + k];
    else
        #pragma omp parallel for
        for(int row = 0; row < _rows; row++)
            for(int k = 0; k < _cols * 3; k += 3)
            {
                if(origin)
                {
                    ((uchar*)_data[0])[(_rows - row - 1)*_cols + k / 3] = img.data[row * img.step + k + 2];
                    ((uchar*)_data[1])[(_rows - row - 1)*_cols + k / 3] = img.data[row * img.step + k + 1];
                    ((uchar*)_data[2])[(_rows - row - 1)*_cols + k / 3] = img.data[row * img.step + k];
                }
                else
                {
                    ((uchar*)_data[0])[row * _cols + k / 3] = img.data[row * img.step + k + 2];
                    ((uchar*)_data[1])[row * _cols + k / 3] = img.data[row * img.step + k + 1];
                    ((uchar*)_data[2])[row * _cols + k / 3] = img.data[row * img.step + k];
                }
            }

    if(_keepIpl)
        _iplimg = img;
}

template<> void					Matrixu::GrayIplImage2Matrix(Mat& img)
{
    //Resize(img->height, img->width, img->nChannels);
    bool origin = false; //img->origin==1;

    if(_depth == 1)
        for(int row = 0; row < _rows; row++)
            for(int k = 0; k < _cols; k++)
                if(origin)
                    ((uchar*)_data[0])[(_rows - row - 1)*_cols + k] = img.data[row * img.step + k];
                else
                    ((uchar*)_data[0])[row * _cols + k] = img.data[row * img.step + k];

}


template<> Matrixu				Matrixu::imResize(float r, float c)
{
    float pr, pc;
    int nr, nc;
    if(c < 0)
    {
        pr = r;
        pc = r;
        nr = (int)(r * _rows);
        nc = (int)(r * _cols);
    }
    else
    {
        pr = r / _rows;
        pc = c / _cols;
        nr = (int)r;
        nc = (int)c;
    }

    Matrixu res((int)(nr), (int)(nc), _depth);
    int type = CV_8UC1;

    for(int k = 0; k < _depth; k++)
    {

        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(nr, nc, type, res._data[k]);

        resize(m1, m3, Size(nc, nr));
    }


    return res;
}

template<> float				Matrixu::ii(const int row, const int col, const int depth) const
{
    return (float)((float*)_iidata[depth])[row * _iipixStep + col];
}
template<> float				Matrixu::dii_dx(uint x, uint y, uint channel)
{
    if(!isInitII()) abortError(__LINE__, __FILE__, "cannot take dii/dx, ii is not init");

    if((x + 1) > (uint)cols() || x < 1) return 0.0f;

    return 0.5f * (ii(y, (x + 1), channel) - ii(y, (x - 1), channel));
}

template<> float				Matrixu::dii_dy(uint x, uint y, uint channel)
{
    if(!isInitII()) abortError(__LINE__, __FILE__, "cannot take dii/dx, ii is not init");

    if((y + 1) > (uint)rows() || y < 1) return 0.0f;

    return 0.5f * (ii((y + 1), x, channel) - ii((y - 1), x, channel));
}

template<> void					Matrixu::initII()
{
    bool err = false;
    _iidata.resize(_depth);
    for(uint k = 0; k < _data.size(); k++)
    {
        _iidataSize = (_rows + 1) * (_cols + 1) * sizeof(float);
        if(_iidata[k] == NULL)
        {
            _iidata[k] = (float*) malloc(_iidataSize);
        }
        if(_iidata[k] == NULL) abortError(__LINE__, __FILE__, "OUT OF MEMORY!");
        _iipixStep = _iidataSize / sizeof(float);

        Mat src(_rows, _cols, CV_8UC1, _data[k]);
        Mat dst(_rows + 1, _cols + 1, CV_32FC1, _iidata[k]);

        integral(src, dst, CV_32F);

        err = err || _data[k] == NULL;
    }
    _ii_init = true;
}

template<> float Matrixu::sumRect(const Rect& rect, int channel) const
{

    // debug checks
    assert(_ii_init);
    assert(rect.x >= 0 && rect.y >= 0 && (rect.y + rect.height) <= _rows
           && (rect.x + rect.width) <= _cols && channel < _depth);
    int maxy = (rect.y + rect.height) * (_cols + 1); // _iipixSteps
    int maxx = rect.x + rect.width;
    int y = rect.y * (_cols + 1);

    float tl = ((float*)_iidata[channel])[y + rect.x];
    float tr = ((float*)_iidata[channel])[y + maxx];
    float br = ((float*)_iidata[channel])[maxy + maxx];
    float bl = ((float*)_iidata[channel])[maxy + rect.x];

    return br + tl - tr - bl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> void					Matrixu::drawRect(Rect rect, int lineWidth, int R, int G, int B)
{
    createIpl();
    Point p1(rect.x, rect.y), p2(rect.x + rect.width, rect.y + rect.height);
    rectangle(_iplimg, p1, p2, CV_RGB(R, G, B), lineWidth);
    IplImage2Matrix(_iplimg);
    freeIpl();
}

template<> void					Matrixu::drawRect(float width, float height, float x, float y, float sc, float th, int lineWidth, int R, int G, int B)
{

    sc = 1.0f / sc;
    th = -th;

    double cth = cos(th) * sc;
    double sth = sin(th) * sc;

    CvPoint p1, p2, p3, p4;

    p1.x = (int)(-cth * width / 2 + sth * height / 2 + width / 2 + x);
    p1.y = (int)(-sth * width / 2 - cth * height / 2 + height / 2 + y);

    p2.x = (int)(cth * width / 2 + sth * height / 2 + width / 2 + x);
    p2.y = (int)(sth * width / 2 - cth * height / 2 + height / 2 + y);

    p3.x = (int)(cth * width / 2 - sth * height / 2 + width / 2 + x);
    p3.y = (int)(sth * width / 2 + cth * height / 2 + height / 2 + y);

    p4.x = (int)(-cth * width / 2 - sth * height / 2 + width / 2 + x);
    p4.y = (int)(-sth * width / 2 + cth * height / 2 + height / 2 + y);

    createIpl();
    line(_iplimg, p1, p2, CV_RGB(R, G, B), lineWidth, CV_AA);
    line(_iplimg, p2, p3, CV_RGB(R, G, B), lineWidth, CV_AA);
    line(_iplimg, p3, p4, CV_RGB(R, G, B), lineWidth, CV_AA);
    line(_iplimg, p4, p1, CV_RGB(R, G, B), lineWidth, CV_AA);
    IplImage2Matrix(_iplimg);
    freeIpl();
}



template<> void					Matrixu::drawEllipse(float height, float width, float x, float y, int lineWidth, int R, int G, int B)
{
    createIpl();
    Point p((int)x, (int)y);
    Size s((int)width, (int)height);
    ellipse(_iplimg, p, s, 0, 0, 365, CV_RGB(R, G, B), lineWidth);
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void					Matrixu::drawEllipse(float height, float width, float x, float y, float startang, float endang, int lineWidth, int R, int G, int B)
{
    createIpl();
    Point p((int)x, (int)y);
    Size s((int)width, (int)height);
    ellipse(_iplimg, p, s, 0, startang, endang, CV_RGB(R, G, B), lineWidth);
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void					Matrixu::drawText(const char* txt, float x, float y, int R, int G, int B)
{
    createIpl();
    Point p((int)x, (int)y);
    putText(_iplimg, txt, p, CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(R, G, B));
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void					Matrixu::warp(Matrixu& res, uint rows, uint cols, float x, float y, float sc, float th, float sr, float phi)
{
    res.Resize(rows, cols, _depth);

    double coeffs[2][3];
    double quad[4][2];

    Mat transform(2, 3, CV_32FC1);

    double cth = cos(th) * sc;
    double sth = sin(th) * sc;

    transform.at<float>(0, 0) = cth;
    transform.at<float>(1, 0) = sth;
    transform.at<float>(0, 1) = -sth;
    transform.at<float>(1, 1) = cth;

    transform.at<float>(0, 2) = x;
    transform.at<float>(1, 2) = y;

    Rect r;
    r.x = (int)x;
    r.y = (int)y;
    r.width = cols;
    r.height = rows;

    int type = CV_8UC1;

    //#pragma omp parallel for
    for(int k = 0; k < _depth; k++)
    {
        Mat m1(_rows, _cols, type, _data[k]);
        Mat m3(rows, cols, type, res._data[k]);
        warpAffine(m1, m3, transform, Size(rows, cols));
    }

}

template<> void					Matrixu::warpAll(uint rows, uint cols, vector<vectorf> params, vector<Matrixu>& res)
{
    res.resize(params[0].size());

    #pragma omp parallel for
    for(int k = 0; k < (int)params[0].size(); k++)
        warp(res[k], rows, cols, params[0][k], params[1][k], params[2][k], params[3][k]);
}
template<> void					Matrixu::computeGradChannels()
{
    float kernel[3] = { -1, 0, 1};

    Mat row(3, 1, CV_32FC1);
    Mat col(1, 3, CV_32FC1);

    row.at<float>(0, 0) = -1;
    row.at<float>(1, 0) = 0;
    row.at<float>(2, 0) = 1;

    col = row.t();

    Size r = _roi;
    r.width -= 3;
    r.height -= 3;

    int type = CV_8UC1;
    Mat m1(_rows, _cols, type, _data[0]);
    Mat m2(_rows, _cols, type, _data[_depth - 2]);
    Mat m3(_rows, _cols, type, _data[_depth - 1]);

    filter2D(m1, m2, CV_8U, row);
    filter2D(m1, m3, CV_8U, col);

}

template<> void					Matrixu::conv2RGB(Matrixu& res)
{
    res.Resize(_rows, _cols, 3);
    for(int k = 0; k < _dataSize; k++)
    {
        ((uchar*)res._data[0])[k] = ((uchar*)_data[0])[k];
        ((uchar*)res._data[1])[k] = ((uchar*)_data[0])[k];
        ((uchar*)res._data[2])[k] = ((uchar*)_data[0])[k];
    }
}
template<> void					Matrixu::conv2BW(Matrixu& res)
{
    res.Resize(_rows, _cols, 1);
    double t;
    for(int k = 0; k < (int)size(); k++)
    {
        t = (double)((uchar*)_data[0])[k];
        t += (double)((uchar*)_data[1])[k];
        t += (double)((uchar*)_data[2])[k];
        ((uchar*)res._data[0])[k] = (uchar)(t / 3.0);
    }

    if(res._keepIpl) res.freeIpl();
}
template<> float				Matrixf::Dot(const Matrixf& x)
{
    assert(this->size() == x.size());
    float sum = 0.0f;
    #pragma omp parallel for reduction(+: sum)
    for(int i = 0; i < (int)size(); i++)
        sum += (*this)(i) * x(i);

    return sum;
}
