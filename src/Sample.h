// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_SAMPLE
#define H_SAMPLE

#include "Matrix.h"
#include "Public.h"


class Sample;


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class Sample
{
public:
    Sample(Matrixu *img, int row, int col, int width = 0, int height = 0, float weight = 1.0);
    Sample()
    {
        _img = NULL;
        _row = _col = _height = _width = 0;
        _weight = 1.0f;
    };
    Sample&				operator= (const Sample& a);

public:
    Matrixu				*_img;
    int					_row, _col, _width, _height;
    float				_weight;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class SampleSet
{
public:
    SampleSet() {};
    SampleSet(const Sample& s)
    {
        _samples.push_back(s);
    };

    int					size() const
    {
        return _samples.size();
    };
    void				push_back(const Sample& s)
    {
        _samples.push_back(s);
    };
    void				push_back(Matrixu *img, int x, int y, int width = 0, int height = 0, float weight = 1.0f);
    void				resize(int i)
    {
        _samples.resize(i);
    };
    void				resizeFtrs(int i);
    float& 				getFtrVal(int sample, int ftr)
    {
        return _ftrVals[ftr](sample);
    };
    float				getFtrVal(int sample, int ftr) const
    {
        return _ftrVals[ftr](sample);
    };
    Sample& 			operator[](const int sample)
    {
        return _samples[sample];
    };
    Sample				operator[](const int sample) const
    {
        return _samples[sample];
    };
    Matrixf				ftrVals(int ftr) const
    {
        return _ftrVals[ftr];
    };
    bool				ftrsComputed() const
    {
        return !_ftrVals.empty() && !_samples.empty() && _ftrVals[0].size() > 0;
    };
    void				clear()
    {
        _ftrVals.clear();
        _samples.clear();
    };


    // densly sample the image in a donut shaped region: will take points inside circle of radius inrad,
    // but outside of the circle of radius outrad.  when outrad=0 (default), then just samples points inside a circle
    void				sampleImage(Matrixu *img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000);
    void				sampleImage(Matrixu *img, uint num, int w, int h);



private:
    vector<Sample>		_samples;
    vector<Matrixf>		_ftrVals; // [ftr][sample]


};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

inline Sample&			Sample::operator= (const Sample& a)
{
    _img	= a._img;
    _row	= a._row;
    _col	= a._col;
    _width	= a._width;
    _height	= a._height;

    return (*this);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void				SampleSet::resizeFtrs(int nftr)
{
    _ftrVals.resize(nftr);
    int nsamp = _samples.size();

    if(nsamp > 0)
        for(int k = 0; k < nftr; k++)
            _ftrVals[k].Resize(1, nsamp);
}

inline void				SampleSet::push_back(Matrixu *img, int x, int y, int width, int height, float weight)
{
    Sample s(img, y, x, width, height, weight);
    push_back(s);
}


#endif