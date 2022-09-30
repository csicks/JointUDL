/***************************************************************************
 *
 * Authors:    Yuxuan Chen
 *
 * Copyright (C) 2022 Pattern Recognition and Bioinformatics Group, Shanghai Jiao Tong University
 *
 * Licensed under the GNU General Public License v3.0 (see LICENSE for details)
 *
 * All comments concerning this program package may be sent to e-mail address 'yxchen11@sjtu.edu.cn'
 ***************************************************************************/

#include <iostream>
#include "mrc.h"
#include "image2D.h"
#include "align.h"
#include "xmippXMD.h"
#include "alignEvaluation.h"

using namespace std;

/** convert every image's value range in an image stack to [0, 255]
 * @return normalized image stack
 */
stackReal<double> rangeOne(const stackReal<double> &stk)
{
    stackReal<double> rr(stk.shape);
    for (int i = 0; i < stk.shape[0]; ++i)
    {
        imageReal<double> img = stk.pieceGet(i);
        double ma = img.max();
        double mi = img.min();
        imageReal<double> r = (img - mi) / (ma - mi) * 255;
        rr.pieceSet(i, r);
    }

    return rr;
}

/** read rotation information generated by UDL/JUDL (for typical XMD files generated by UDL)
 * @return list containing rotation angle for each image
 */
std::vector<double> readParamsUDL(const std::string &path)
{
    std::ifstream in;
    in.open(path, std::ios::in);
    if (!in)
        throw baseException("Error: Unable to open txt file!");
    std::string line;
    std::vector<double> r;
    int row = 0;
    while (getline(in, line))
    {
        if (row < 9)
        {
            ++row;
            continue;
        }
        std::istringstream sin(line);
        std::vector<std::string> fields;
        std::string field;
        while (getline(sin, field, '\t'))
        {
            fields.emplace_back(field);
        }
        double number = atof(fields[3].c_str());
        r.push_back(number);
    }
    return r;
}

/** read rotation information generated by UDL/JUDL (for simple rotation angles in TXT file)
 * @return list containing rotation angle for each image
 */
// std::vector<double> readParamsUDL(const std::string &path)
// {
//     std::ifstream in;
//     in.open(path, std::ios::in);
//     if (!in)
//         throw baseException("Error: Unable to open txt file!");
//     std::string line;
//     std::vector<double> r;
//     while (getline(in, line))
//     {
//         std::istringstream sin(line);
//         double number;
//         sin >> number;
//         r.push_back(number);
//     }
//     return r;
// }

/** estimate translations according to rotation informations
 * @return final transformation parameters including rotation and translations
 */
phi rotateShift(double angle, imageReal<double> imgX, imageReal<double> imgC)
{
    return angleAll(angle, imgX, imgC);
}

/** estimate translations for the whole dataset and generate average image
 */
void refinementUDL(stackReal<double> &stk, imageReal<double> &ref, std::vector<double> angles, imageReal<double> &sref)
{
    int N = stk.shape[0];
    phi p;
    double corr = 0;
    for (int i = 0; i < stk.shape[0]; ++i)
    {
        imageReal<double> piece = stk.pieceGet(i);
        p = rotateShift(angles[i], ref, piece);
        imageReal<double> imgAl = applyPhi(piece, p);
        corr += correntropy(imgAl, ref);
        sref = sref + imgAl;
        stk.pieceSet(i, imgAl);
    }
    sref = sref / stk.shape[0];
    corr /= stk.shape[0];
    std::cout << corr << std::endl;
}

/** main function for evaluation. Metrics including correntropy between average images of two equal parts and
 * four equal parts are calculated.
 */
void evaluateRefImageUDL(const std::string &dataPath, const std::string &txtPath)
{
    stackReal<double> data = readData(dataPath);
    data = rangeOne(data);
    std::vector<double> txt = readParamsUDL(txtPath);
    imageReal<double> ref = data.pieceGet(0);
    // avoid influence of bias on similarity calculation
    double angle0 = txt[0];
    phi pref = {angle0, 0, 0};
    ref = applyPhi(ref, pref);

    int shape[2] = {ref.shape[0], ref.shape[1]};
    imageReal<double> sref(shape);

    clock_t st = clock();
    refinementUDL(data, ref, txt, sref);
    clock_t et = clock();
    std::cout << "Align time:" << (double)(et - st) / CLOCKS_PER_SEC << std::endl;
    mrcFile mrc_new = mrcFile();
    stackReal<double> stk = image2Stack(sref);

    // two parts division
    stackReal<double> stk12 = data.pieces(0, int(data.shape[0] / 2));
    stackReal<double> stk22 = data.pieces(int(data.shape[0] / 2) + 1, data.shape[0] - 1);
    imageReal<double> avg12 = stk12.stackAvg();
    imageReal<double> avg22 = stk22.stackAvg();
    double corr = correntropy(avg12, avg22);
    cout << corr << endl;

    // four parts division
    stackReal<double> stk14 = data.pieces(0, int(data.shape[0] / 4));
    stackReal<double> stk24 = data.pieces(int(data.shape[0] / 4) + 1, int(data.shape[0] / 2) + 1);
    stackReal<double> stk34 = data.pieces(int(data.shape[0] / 2) + 1, int(data.shape[0] * 3 / 4) + 1);
    stackReal<double> stk44 = data.pieces(int(data.shape[0] * 3 / 4) + 1, data.shape[0] - 1);
    imageReal<double> avg14 = stk14.stackAvg();
    imageReal<double> avg24 = stk24.stackAvg();
    imageReal<double> avg34 = stk34.stackAvg();
    imageReal<double> avg44 = stk44.stackAvg();
    double corr12 = correntropy(avg14, avg24);
    double corr13 = correntropy(avg14, avg34);
    double corr14 = correntropy(avg14, avg44);
    double corr23 = correntropy(avg24, avg34);
    double corr24 = correntropy(avg24, avg44);
    double corr34 = correntropy(avg34, avg44);
    cout << (corr12 + corr13 + corr14 + corr23 + corr24 + corr34) / 6 << endl;

    mrc_new.setData(stk);
    mrc_new.write("align_ref.mrcs");
    mrc_new.setData(data);
    mrc_new.write("align.mrcs");
}

int main(int argc, char *argv[])
{
    string dataPath = "/path/to/mrcs file";
    string txtPath = "/path/to/output txt file generated by UDL/JUDL";
    evaluateRefImageUDL(dataPath, txtPath);
    return 0;
}

/** code for Fourier-based method evaluation and area-based method evaluation is attached below, which is nearly the same as code above. Comments are ignored.
 */
// #include <iostream>
// #include "mrc.h"
// #include "image2D.h"
// #include "align.h"
// #include "xmippXMD.h"
// #include "alignEvaluation.h"
// using namespace std;

// stackReal<double> rangeOne(const stackReal<double> &stk)
// {
//     stackReal<double> rr(stk.shape);
//     for (int i = 0; i < stk.shape[0]; ++i)
//     {
//         imageReal<double> img = stk.pieceGet(i);
//         double ma = img.max();
//         double mi = img.min();
//         imageReal<double> r = (img - mi) / (ma - mi) * 255;
//         rr.pieceSet(i, r);
//     }

//     return rr;
// }

// void refinementOld(stackReal<double> &stk, imageReal<double> &ref, imageReal<double> &sref)
// {
//     int N = stk.shape[0];
//     phi p;
//     double corr = 0;
//     for (int i = 0; i < stk.shape[0]; ++i)
//     {
//         imageReal<double> piece = stk.pieceGet(i);
//         // choose Fourier-based method alignShape or area-based method alignFFTX
//         // p = alignShape(ref, piece, stk.shape[1] / 20);
//         p = alignFFTX(ref, piece);
//         imageReal<double> imgAl = applyPhi(piece, p);
//         corr += correntropy(imgAl, ref);
//         sref = sref + imgAl;
//         stk.pieceSet(i, imgAl);
//     }
//     sref = sref / stk.shape[0];
//     corr /= stk.shape[0];
//     std::cout << corr << std::endl;
// }

// void evaluateRefImageOld(const std::string &dataPath)
// {
//     stackReal<double> data = readData(dataPath);
//     data = rangeOne(data);
//     imageReal<double> ref = data.pieceGet(0);
//     int shape[2] = {ref.shape[0], ref.shape[1]};
//     imageReal<double> sref(shape);

//     clock_t st = clock();
//     refinementOld(data, ref, sref);
//     clock_t et = clock();
//     std::cout << "Align time:" << (double)(et - st) / CLOCKS_PER_SEC << std::endl;
//     mrcFile mrc_new = mrcFile();
//     stackReal<double> stk = image2Stack(sref);

//     // two parts division
//     stackReal<double> stk12 = data.pieces(0, int(data.shape[0] / 2));
//     stackReal<double> stk22 = data.pieces(int(data.shape[0] / 2) + 1, data.shape[0] - 1);
//     imageReal<double> avg12 = stk12.stackAvg();
//     imageReal<double> avg22 = stk22.stackAvg();
//     double corr = correntropy(avg12, avg22);
//     cout << corr << endl;

//     // four parts division
//     stackReal<double> stk14 = data.pieces(0, int(data.shape[0] / 4));
//     stackReal<double> stk24 = data.pieces(int(data.shape[0] / 4) + 1, int(data.shape[0] / 2) + 1);
//     stackReal<double> stk34 = data.pieces(int(data.shape[0] / 2) + 1, int(data.shape[0] * 3 / 4) + 1);
//     stackReal<double> stk44 = data.pieces(int(data.shape[0] * 3 / 4) + 1, data.shape[0] - 1);
//     imageReal<double> avg14 = stk14.stackAvg();
//     imageReal<double> avg24 = stk24.stackAvg();
//     imageReal<double> avg34 = stk34.stackAvg();
//     imageReal<double> avg44 = stk44.stackAvg();

//     double corr12 = correntropy(avg14, avg24);
//     double corr13 = correntropy(avg14, avg34);
//     double corr14 = correntropy(avg14, avg44);
//     double corr23 = correntropy(avg24, avg34);
//     double corr24 = correntropy(avg24, avg44);
//     double corr34 = correntropy(avg34, avg44);
//     cout << (corr12 + corr13 + corr14 + corr23 + corr24 + corr34) / 6 << endl;

//     mrc_new.setData(stk);
//     mrc_new.write("align_ref.mrcs");
//     mrc_new.setData(data);
//     mrc_new.write("align.mrcs");
// }

// int main(int argc, char *argv[])
// {
//     string dataPath = "/path/to/mrcs file";
//     evaluateRefImageOld(dataPath);
//     return 0;
// }