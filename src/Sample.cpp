// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Sample.h"

Sample::Sample(Matrixu *img, int row, int col, int width, int height, float weight)
{
    _img	= img;
    _row	= row;
    _col	= col;
    _width	= width;
    _height	= height;
    _weight = weight;
}



void		SampleSet::sampleImage(Matrixu *img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
{
    int rowsz = img->rows() - h - 1;
    int colsz = img->cols() - w - 1;
    float inradsq = inrad * inrad;
    float outradsq = outrad * outrad;
    int dist;

    uint minrow = max(0, (int)y - (int)inrad);
    uint maxrow = min((int)rowsz - 1, (int)y + (int)inrad);
    uint mincol = max(0, (int)x - (int)inrad);
    uint maxcol = min((int)colsz - 1, (int)x + (int)inrad);

    //fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d => %d\n",inrad,minrow,maxrow,mincol,maxcol, (maxrow-minrow+1)*(maxcol-mincol+1) );

//	Mat m = img->getIpl();
//	Mat vis;
//	m.copyTo(vis);


    _samples.resize((maxrow - minrow + 1) * (maxcol - mincol + 1));
    int i = 0;

    float prob = ((float)(maxnum)) / _samples.size();

    //#pragma omp parallel for
    for(int r = minrow; r <= (int)maxrow; r++)
        for(int c = mincol; c <= (int)maxcol; c++)
        {
            dist = (y - r) * (y - r) + (x - c) * (x - c);
            if(randfloat() < prob && dist < inradsq && dist >= outradsq)
            {
                _samples[i]._img = img;
                _samples[i]._col = c;
                _samples[i]._row = r;
                _samples[i]._height = h;
                _samples[i]._width = w;

//cv::rectangle(vis, Point(c, r), Point(w - c, h- r), Scalar(255), 1);
                i++;
            }
        }

//cv::imshow("Samples", vis);


    fprintf(stderr, "%d %d\n", i, maxnum);
    _samples.resize(min(i, maxnum));

}

void		SampleSet::sampleImage(Matrixu *img, uint num, int w, int h)
{
    int rowsz = img->rows() - h - 1;
    int colsz = img->cols() - w - 1;

    _samples.resize(num);
    //#pragma omp parallel for
    for(int i = 0; i < (int)num; i++)
    {
        _samples[i]._img = img;
        _samples[i]._col = randint(0, colsz);
        _samples[i]._row = randint(0, rowsz);
        _samples[i]._height = h;
        _samples[i]._width = w;
    }
}
