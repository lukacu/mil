// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef TRACKER_PUBLIC
#define TRACKER_PUBLIC

#include "OnlineBoost.h"
#include "Public.h"

class TrackerParams
{
public:
    TrackerParams();

    uint			_negnumtrain, _init_negnumtrain; // # negative samples to use during training, and init
    float			_posradtrain, _init_postrainrad; // radius for gathering positive instances
    uint			_posmaxtrain;					// max # of pos to train with
    bool			_debugv;						// displays response map during tracking [kinda slow, but help in debugging]
    vectorf			_initstate;						// [x,y,scale,orientation] - note, scale and orientation currently not used
    bool			_useLogR;						// use log ratio instead of probabilities (tends to work much better)
    bool			_initWithFace;					// initialize with the OpenCV tracker rather than _initstate
    bool			_disp;							// display video with tracker state (colored box)

    uint			_srchwinsz;						// size of search window
    uint			_negsamplestrat;				// [0] all over image [1 - default] close to the search window

};

class Tracker
{
public:
    Tracker()
    {
        _clf = NULL;
    };
    ~Tracker()
    {
        if(_clf != NULL) delete _clf;
    };

    double			track_frame(Matrixu& frame); // track object in a frame;  requires init() to have been called.
    bool			init(Matrixu& frame, TrackerParams p, ClfStrongParams *clfparams);
    cv::Rect		getRectangle();


private:
    ClfStrong			*_clf;
    vectorf				_curState;
    TrackerParams	_trparams;
    ClfStrongParams		*_clfparams;
    int					_cnt;
};



#endif



