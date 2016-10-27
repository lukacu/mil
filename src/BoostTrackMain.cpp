// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Matrix.h"
#include "ImageFtr.h"
#include "Tracker.h"
#include "Public.h"

#include <trax/opencv.hpp>

ClfStrongParams* parseParameters(trax::Properties& param, TrackerParams& trparams)
{

    ClfStrongParams			*clfparams;
    string type = param.get("type", "mil");

    // strong model
    if(type == "mil")    // MILTrack
    {
        clfparams = new ClfMilBoostParams();
        ((ClfMilBoostParams*)clfparams)->_numSel = param.get("num_sel", 50);
        ((ClfMilBoostParams*)clfparams)->_numFeat = param.get("num_feat", 250);
        trparams._posradtrain							= param.get("pos_rad_train", 4.0f);
        trparams._negnumtrain							= param.get("neg_num_train", 65);
    }
    else if(type == "oba1")      // OBA1
    {
        clfparams = new ClfAdaBoostParams();
        ((ClfAdaBoostParams*)clfparams)->_numSel		= param.get("num_sel", 50);
        ((ClfAdaBoostParams*)clfparams)->_numFeat		= param.get("num_feat", 250);
        trparams._posradtrain							= param.get("pos_rad_train", 1.0f);
        trparams._negnumtrain							= param.get("neg_num_train", 65);
    }
    else if(type == "oba5")      // OBA5
    {
        clfparams = new ClfAdaBoostParams();
        ((ClfAdaBoostParams*)clfparams)->_numSel		= param.get("num_sel", 50);
        ((ClfAdaBoostParams*)clfparams)->_numFeat		= param.get("num_feat", 250);
        trparams._posradtrain							= param.get("pos_rad_train", 4.0f);
        trparams._negnumtrain							= param.get("neg_num_train", 65);
    }
    else
    {
        abortError(__LINE__, __FILE__, "Error: invalid classifier choice.");
    }

    // tracking parameters
    trparams._init_negnumtrain = param.get("init_neg_num_train", 65);
    trparams._init_postrainrad = param.get("init_pos_rad_train", 3.0f);

    trparams._srchwinsz		= param.get("search_window", 25);
    trparams._negsamplestrat = param.get("negsamplestrat", 1);
    trparams._debugv		= false;

    clfparams->_ftrParams = new HaarFtrParams();
    cout << clfparams->_ftrParams->_width << " " << clfparams->_ftrParams->_height << endl;
    return clfparams;
}

int main(int argc, char * argv[])
{

    trax::Image img;
    trax::Region reg;
    cv::Mat image, gray;
    cv::Rect rectangle;

    trax::Server handle(trax::Configuration(TRAX_IMAGE_PATH | TRAX_IMAGE_MEMORY | TRAX_IMAGE_BUFFER, TRAX_REGION_RECTANGLE), trax_no_log);

    ClfStrongParams	*clfparams;
    TrackerParams trparams;
    float scale = 1.f;

    Matrixu m;
    Tracker* tracker = NULL;

    bool run = 1;

    while(run)
    {

        trax::Properties prop;
        trax::Region status;

        int tr = handle.wait(img, reg, prop);

        // There are two important commands. The first one is TRAX_INITIALIZE that tells the
        // tracker how to initialize.
        if(tr == TRAX_INITIALIZE)
        {

            if(tracker)
            {
                delete tracker;
                tracker = NULL;
                delete clfparams->_ftrParams;
                delete clfparams;
                clfparams = NULL;
            }

            clfparams = parseParameters(prop, trparams);

            image = trax::image_to_mat(img);

            if(image.channels() == 3)
                cv::cvtColor(image, gray, CV_BGR2GRAY);
            else
                gray = image;

            float x, y, width, height;
            reg.get(&x, &y, &width, &height);

            trparams._initstate[0]	= x;
            trparams._initstate[1]	= y;
            trparams._initstate[2]	= width;
            trparams._initstate[3]	= height;

            m.Resize(gray.rows, gray.cols, 1);
            m.GrayIplImage2Matrix(gray);

            tracker = new Tracker();
            if(!tracker->init(m, trparams, clfparams))
            {
                abortError(__LINE__, __FILE__, "Unable to initialize tracker");
            }
            status = reg;

        }
        else
            // The second one is TRAX_FRAME that tells the tracker what to process next.
            if(tr == TRAX_FRAME)
            {

                image = trax::image_to_mat(img);

                if(image.channels() == 3)
                    cv::cvtColor(image, gray, CV_BGR2GRAY);
                else
                    gray = image;

                m.GrayIplImage2Matrix(gray);

                tracker->track_frame(m);

                cv::Rect bb = tracker->getRectangle();

                status = trax::rect_to_region(bb);

            }
        // Any other command is either TRAX_QUIT or illegal, so we exit.
            else
            {
                break;
            }

        handle.reply(status, trax::Properties());

    }

    if(tracker)
    {
        delete tracker;
        tracker = NULL;
        delete clfparams->_ftrParams;
        delete clfparams;
        clfparams = NULL;
    }


    return EXIT_SUCCESS;

}

