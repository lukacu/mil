// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"

bool Tracker::init(Matrixu& frame, TrackerParams p, ClfStrongParams *clfparams)
{
    static Matrixu *img;

    img = &frame;
    frame.initII();

    clfparams->_ftrParams->_width = (uint)p._initstate[2];
    clfparams->_ftrParams->_height = (uint)p._initstate[3];

    _clf = ClfStrong::makeClf(clfparams);
    _curState.resize(4);
    for(int i = 0; i < 4; i++) _curState[i] = p._initstate[i];
    SampleSet posx, negx;

    fprintf(stderr, "Initializing Tracker..\n");

    // sample positives and negatives from first frame
    posx.sampleImage(img, (uint)_curState[0], (uint)_curState[1], (uint)_curState[2], (uint)_curState[3], p._init_postrainrad);
    negx.sampleImage(img, (uint)_curState[0], (uint)_curState[1], (uint)_curState[2], (uint)_curState[3], 2.0f * p._srchwinsz, (1.5f * p._init_postrainrad), p._init_negnumtrain);

    if(posx.size() < 1 || negx.size() < 1) return false;

    // train
    _clf->update(posx, negx);
    negx.clear();

    img->FreeII();

    _trparams = p;
    _clfparams = clfparams;
    _cnt = 0;
    return true;
}

double Tracker::track_frame(Matrixu& frame)
{
    static SampleSet posx, negx, detectx;
    static vectorf prob;
    static vectori order;
    static Matrixu *img;

    double resp;

    img = &frame;
    frame.initII();

    // run current clf on search window
    detectx.sampleImage(img, (uint)_curState[0], (uint)_curState[1], (uint)_curState[2], (uint)_curState[3], (float)_trparams._srchwinsz);
    prob = _clf->classify(detectx, _trparams._useLogR);

    // find best location
    int bestind = max_idx(prob);
    resp = prob[bestind];

    _curState[1] = (float)detectx[bestind]._row;
    _curState[0] = (float)detectx[bestind]._col;

    // train location clf (negx are randomly selected from image, posx is just the current tracker location)

    if(_trparams._negsamplestrat == 0)
        negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
    else
        negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3],
                         (1.5f * _trparams._srchwinsz), _trparams._posradtrain + 5, _trparams._negnumtrain);

    if(_trparams._posradtrain == 1)
        posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
    else
        posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

    _clf->update(posx, negx);

    /*if( _trparams._debugv ) {
        for( int j=0; j<negx.size(); j++ )
            framedisp.drawEllipse(1,1,(float)negx[j]._col,(float)negx[j]._row,1,255,0,255);
    }*/

    // clean up
    img->FreeII();
    posx.clear();
    negx.clear();
    detectx.clear();

    _cnt++;

    return resp;
}

cv::Rect Tracker::getRectangle()
{
    Rect rect;
    rect.x = _curState[0];
    rect.y = _curState[1];
    rect.width = _curState[2];
    rect.height = _curState[3];

    return rect;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
TrackerParams::TrackerParams()
{
    _negnumtrain	= 15;
    _posradtrain	= 1;
    _posmaxtrain	= 100000;
    _init_negnumtrain = 1000;
    _init_postrainrad = 3;
    _initstate.resize(4);
    _debugv			= false;
    _useLogR		= true;
    _disp			= true;
    _srchwinsz		= 30;
    _initstate.resize(4);
    _negsamplestrat	= 1;
}


