using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using System.Globalization;

namespace OpenCvYolo3
{
    /// <summary>
    /// OpenCvS V4 with YOLO v3

    /// YOLO
    /// https://pjreddie.com/darknet/yolo/
    /// </summary>

    public class Yolo3Android : MonoBehaviour
    {
        #region
        private string cfg = "yolov3.cfg";
        private string weight = "yolov3.weights";
        private string names = "coco.names";
        private string image = "kite.jpg";

        const float threshold = 0.24f;       //for confidence 
        const float nmsThreshold = 0.24f;    //threshold for nms

        //random assign color to each label
        private static Scalar[] Colors;

        //get labels from coco.names
        private static string[] Labels;
        #endregion

        void Start()
        {
            cfg = Utils.getFilePath("dnn/" + cfg);
            weight = Utils.getFilePath("dnn/" + weight);
            names = Utils.getFilePath("dnn/" + names);
            image = Utils.getFilePath("dnn/" + image);
            
            Labels = readClassNames(names).ToArray();
            Colors = Enumerable
                .Repeat(false, Labels.Length)
                .Select(x => new Scalar((int)UnityEngine.Random.Range(0f, 256f), (int)UnityEngine.Random.Range(0f, 256f), (int)UnityEngine.Random.Range(0f, 256f)))
                .ToArray();
  
            ObjectDetection();
        }

        void ObjectDetection()
        {
            // If true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            Utils.setDebugMode(true);

            Mat img = Imgcodecs.imread(image);
            if (img.empty())
            {
                Debug.LogError("Image " + image + " is not loaded.");
                img = new Mat(424, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
            }


            Net net = null;

            if (string.IsNullOrEmpty(cfg) || string.IsNullOrEmpty(weight))
            {
                Debug.LogError(cfg + " or " + weight + " is not loaded.");
            }
            else
            {   
                //load model and config
                net = Dnn.readNet(weight, cfg);
            }

            if (net == null)
            {

                Imgproc.putText(img, "model file is not loaded.", new Point(5, img.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2, Imgproc.LINE_AA, false);
                Imgproc.putText(img, "Please read console message.", new Point(5, img.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2, Imgproc.LINE_AA, false);

            }
            else
            {
                //setting blob, size can be:320/416/608
                //opencv blob setting can check here https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection
                Mat blob = Dnn.blobFromImage(img, 1.0 / 255, new Size(416, 416), new Scalar(0), false, false);

                //input data
                net.setInput(blob);

                //get output layer name
                List<string> outNames = net.getUnconnectedOutLayersNames();
                //create mats for output layer
                List<Mat> outs = outNames.Select(_ => new Mat()).ToList();

                #region forward model
                TickMeter tm = new TickMeter();
                tm.start();

                net.forward(outs, outNames);

                tm.stop();
                Debug.Log("Runtime: " + tm.getTimeMilli() + " ms");
                #endregion

                //get result from all output
                GetResult(outs, img, threshold, nmsThreshold);
            }

            // Show Image
            Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
            Texture2D texture = new Texture2D(img.cols(), img.rows(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(img, texture);
            gameObject.GetComponent<Renderer>().material.mainTexture = texture;
            Utils.setDebugMode(false);
        }

        /// <summary>
        /// Get result form all output
        /// </summary>
        /// <param name="output"></param>
        /// <param name="image"></param>
        /// <param name="threshold"></param>
        /// <param name="nmsThreshold">threshold for nms</param>
        /// <param name="nms">Enable Non-maximum suppression or not</param>
        private static void GetResult(IEnumerable<Mat> output, Mat image, float threshold, float nmsThreshold, bool nms = true)
        {
            //for nms
            List<int> classIds = new List<int>();
            List<float> confidences = new List<float>();
            List<float> probabilities = new List<float>();
            List<Rect2d> boxes = new List<Rect2d>();

            var w = image.width();
            var h = image.height();
            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability 
            */
            const int prefix = 5;   //skip 0~4

            foreach (Mat prob in output)
            {
                for (int i = 0; i < prob.rows(); i++)
                {
                    var confidence = (float)prob.get(i, 4)[0];
                    if (confidence > threshold)
                    {
                        //get classes probability
                        Core.MinMaxLocResult minAndMax = Core.minMaxLoc(prob.row(i).colRange(prefix, prob.cols()));
                        int classes = (int)minAndMax.maxLoc.x;
                        var probability = (float)prob.get(i, classes + prefix)[0];

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            float centerX = (float)prob.get(i, 0)[0] * w;
                            float centerY = (float)prob.get(i, 1)[0] * h;
                            float width = (float)prob.get(i, 2)[0] * w;
                            float height = (float)prob.get(i, 3)[0] * h;

                            if (!nms)
                            {
                                // draw result (if don't use NMSBoxes)
                                Draw(image, classes, confidence, probability, centerX, centerY, width, height);
                                continue;
                            }

                            //put data to list for NMSBoxes
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(centerX, centerY, width, height));
                        }
                    }
                }
            }

            if (!nms) return;

            //using non-maximum suppression to reduce overlapping low confidence box
            MatOfRect2d bboxes = new MatOfRect2d();
            MatOfFloat scores = new MatOfFloat();
            MatOfInt indices = new MatOfInt();

            bboxes.fromList(boxes);
            scores.fromList(probabilities);


            Dnn.NMSBoxes(bboxes, scores, threshold, nmsThreshold, indices);

            int[] indicesA = indices.toArray();

            foreach (var i in indicesA)
            {
                var box = boxes[i];
                Draw(image, classIds[i], confidences[i], probabilities[i], box.x, box.y, box.width, box.height);
            }

        }

        /// <summary>
        /// Draw result to image
        /// </summary>
        /// <param name="image"></param>
        /// <param name="classes"></param>
        /// <param name="confidence"></param>
        /// <param name="probability"></param>
        /// <param name="centerX"></param>
        /// <param name="centerY"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        private static void Draw(Mat image, int classes, float confidence, float probability, double centerX, double centerY, double width, double height)
        {
            //label formating
            NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;
            nfi.PercentDecimalDigits = 2;
            string label = Labels[classes] + " " + probability.ToString("P", nfi);

            //draw result
            double x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2; //avoid left side over edge
            Imgproc.rectangle(image, new Point(x1, centerY - height / 2), new Point(centerX + width / 2, centerY + height / 2), Colors[classes], 2);

            int[] baseline = new int[1];
            var textSize = Imgproc.getTextSize(label, 4, 0.5, 1, baseline);
            Imgproc.rectangle(image, new OpenCVForUnity.CoreModule.Rect(new Point(x1, centerY - height / 2 - textSize.height - baseline[0]),
                new Size(textSize.width, textSize.height + baseline[0])), Colors[classes], Core.FILLED);

            double mean = 0;
            for (int i = 0; i < 4; i++)
            {
                mean += Colors[classes].val[i];
            }
            mean = mean / 4;
            
            Scalar textColor = mean < 70 ? new Scalar(255,255,255,255) : new Scalar(0);
            Imgproc.putText(image, label, new Point(x1, centerY - height / 2 - baseline[0]), Imgproc.FONT_HERSHEY_TRIPLEX, 0.5, textColor);
        }

        /// <summary>
        /// Reads the class names.
        /// </summary>
        /// <returns>The class names.</returns>
        /// <param name="filename">Filename.</param>
        private List<string> readClassNames(string filename)
        {
            List<string> classNames = new List<string>();

            System.IO.StreamReader cReader = null;
            try
            {
                cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

                while (cReader.Peek() >= 0)
                {
                    string name = cReader.ReadLine();
                    classNames.Add(name);
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError(ex.Message);
                return null;
            }
            finally
            {
                if (cReader != null)
                    cReader.Close();
            }

            return classNames;
        }

    }
}
