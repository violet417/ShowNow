/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.shownow1.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.shownow1.MainActivity;
import org.tensorflow.lite.examples.shownow1.env.Logger;
import org.tensorflow.lite.examples.shownow1.env.Utils;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

public class YoloV5Classifier implements Classifier {
    public static YoloV5Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized,
            final int inputSize)
            throws IOException {
        final YoloV5Classifier d = new YoloV5Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                d.nnapiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    d.nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(d.nnapiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                GpuDelegate.Options gpu_options = new GpuDelegate.Options();
                gpu_options.setPrecisionLossAllowed(true); // It seems that the default is true
                gpu_options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                d.gpuDelegate = new GpuDelegate(gpu_options);
                options.addDelegate(d.gpuDelegate);
            }
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.INPUT_SIZE = inputSize;
        d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];

        d.output_box = (int) ((Math.pow((inputSize / 32), 2) + Math.pow((inputSize / 16), 2) + Math.pow((inputSize / 8), 2)) * 3);
        if (d.isModelQuantized){
            Tensor inpten = d.tfLite.getInputTensor(0);
            d.inp_scale = inpten.quantizationParams().getScale();
            d.inp_zero_point = inpten.quantizationParams().getZeroPoint();
            Tensor oupten = d.tfLite.getOutputTensor(0);
            d.oup_scale = oupten.quantizationParams().getScale();
            d.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }

        int[] shape = d.tfLite.getOutputTensor(0).shape();
        int numClass = shape[shape.length - 1] - 5;
        d.numClass = numClass;
        d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel);
        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }
    @Override
    public void enableStatLogging(final boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        tfLite.close();
        tfLite = null;
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
            nnapiDelegate = null;
        }
        tfliteModel = null;
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
    }

    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    public void useCPU() {
        recreateInterpreter();
    }

    public void useNNAPI() {
        nnapiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnapiDelegate);
        recreateInterpreter();
    }

    @Override
    public float getObjThresh() {
        return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
    }

    private static final Logger LOGGER = new Logger();

    // Float model
    private final float IMAGE_MEAN = 0;

    private final float IMAGE_STD = 255.0f;

    //config yolo
    private int INPUT_SIZE = -1;

    private  int output_box;

    private static final float[] XYSCALE = new float[]{1.2f, 1.1f, 1.05f};

    private static final int NUM_BOXES_PER_BLOCK = 3;

    // Number of threads in the java app
    private static final int NUM_THREADS = 1;
    private static boolean isNNAPI = false;
    private static boolean isGPU = false;

    private boolean isModelQuantized;

    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;
    /** holds an nnapi delegate */
    NnApiDelegate nnapiDelegate = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;

    private ByteBuffer imgData;
    private ByteBuffer outData;

    private Interpreter tfLite;
    private float inp_scale;
    private int inp_zero_point;
    private float oup_scale;
    private int oup_zero_point;
    private int numClass;
    private YoloV5Classifier() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer_ = convertBitmapToByteBuffer(bitmap);

        Map<Integer, Object> outputMap = new HashMap<>();
        HashMap<String,String> nameMap = new HashMap<String,String>(){{
            put("001", "베라민트초코우유");
            put("002", "베라베리베리스트로베리우유");
            put("003", "베라이상한나라의솜사탕우유");
            put("004", "베라쿠키앤크림우유");
            put("005", "K그램B맥주레몬보드카병");
            put("006", "OM그램캔");
            put("007", "강릉커피너티크림라떼");
            put("008", "고길동에일맥주캔");
            put("009", "고티카빈티지라떼");
            put("010", "고티카빈티지블랙");
            put("011", "고티카빈티지스위트아메리카노");
            put("012", "곰표썸머에일캔");
            put("013", "광동제약비타500");
            put("014", "구스아일랜드312얼반위트");
            put("015", "구스아일랜드덕덕구스캔");
            put("016", "구스아일랜드아이피에이캔");
            put("017", "기네스드레프트캔");
            put("018", "기네스엑스트라스타우드캔");
            put("019", "기린이치방시보리캔");
            put("020", "까스활명수");
            put("021", "나랑드사이다캔245");
            put("022", "남양초코에몽250ML");
            put("023", "네스퀵초코");
            put("024", "농심웰치스포도355ml");
            put("025", "닥터유단백질바나나");
            put("026", "닥터유단백질초코");
            put("027", "닥터유프로단백질초코250ml");
            put("028", "닥터캡슐프로텍트베리믹스");
            put("029", "닥터캡슐프로텍트사과");
            put("030", "닥터캡슐프로텍트플레인");
            put("031", "더단백초코");
            put("032", "더단백카라멜");
            put("033", "더단백커피");
            put("034", "덴마크드링킹요구르트딸기");
            put("035", "덴마크드링킹요구르트사과");
            put("036", "덴마크드링킹요구르트샤인머스켓");
            put("037", "덴마크드링킹요구르트플레인");
            put("038", "덴마크딸기딸기");
            put("039", "덴마크민트초코");
            put("040", "덴마크바나나우유");
            put("041", "덴마크얼라이브망고");
            put("042", "덴마크얼라이브머스캣청포도");
            put("043", "덴마크얼라이브블러드오렌지");
            put("044", "덴마크얼라이브스위트자몽");
            put("045", "덴마크초코초코");
            put("046", "델몬트콜드오렌지");
            put("047", "델몬트콜드포도");
            put("048", "동아데미소다애플250ML");
            put("049", "동아데미소다오렌지250ML");
            put("050", "동아오츠카데자와로얄밀크티500ML");
            put("051", "동아포카리스웨트캔240ML");
            put("052", "라떼니스타카라멜라떼");
            put("053", "라떼니스타크리미라떼");
            put("054", "랩노쉬프로틴마일드라떼350ml");
            put("055", "랩노쉬프로틴마일드카카오350ml");
            put("056", "레쓰비마일드커피");
            put("057", "레쓰비카페타임라떼");
            put("058", "레쓰비카페타임헤이즐넛라떼");
            put("059", "레츠프레시투데이캔");
            put("060", "로스터리슈크림라떼");
            put("061", "로스터리아메리카노");
            put("062", "로스터리에스프레소벨벳라떼");
            put("063", "롯데립톤아이스티복숭아500ML");
            put("064", "롯데게토레이240ML");
            put("065", "롯데델몬트알로에400ML");
            put("066", "롯데델몬트오렌지100_400ML");
            put("067", "롯데마운틴듀250ML");
            put("068", "롯데밀키스250ML");
            put("069", "롯데밀키스340ML");
            put("070", "롯데펩시콜라250ML");
            put("071", "롯데펩시콜라600ML");
            put("072", "롱보드아일랜드라거캔");
            put("073", "마이카페라떼마일드");
            put("074", "마이카페라떼마일드로어슈거");
            put("075", "마이카페라떼아이스크림믹스라떼");
            put("076", "마이카페라떼카라멜마끼아또");
            put("077", "매일두유99.9");
            put("078", "매일두유검은콩");
            put("079", "매일두유오리지널");
            put("080", "매일소화가잘되는우유오리지널락토프리");
            put("081", "매일우유속에딸기과즙");
            put("082", "매일허쉬초콜릿드링크쿠키앤크림");
            put("083", "맥스웰하우스마스터오리지날블랙");
            put("084", "맥스웰하우스마스터카페라떼");
            put("085", "맥스웰하우스콜롬비아나스위트아메리카노");
            put("086", "모구모구딸기");
            put("087", "모구모구리치맛");
            put("088", "모구모구복숭아향");
            put("089", "모구모구요거트");
            put("090", "몬스터에너지그린355ML");
            put("091", "몰슨캐네디언캔");
            put("092", "미닛메이드오렌지350ML");
            put("093", "바리스타로어슈거에스프레소라떼");
            put("094", "바리스타마다가스카바닐라빈라떼");
            put("095", "바리스타모카프레소");
            put("096", "바리스타벨지엄쇼콜라모카");
            put("097", "바리스타스모키로스팅라떼");
            put("098", "바리스타플라넬드립라떼");
            put("099", "바이오그레놀라");
            put("100", "바이오초코링프로틴볼");
            put("101", "박카스");
            put("102", "박카스디카페인");
            put("103", "버드와이저캔");
            put("104", "블루문캔");
            put("105", "비요뜨링크");
            put("106", "비요뜨초코팝");
            put("107", "비타오백");
            put("108", "빅웨이브골든에일캔");
            put("109", "빙그레딸기맛우유");
            put("110", "빙그레바나나맛우유");
            put("111", "빙그레바나나맛우유라이트");
            put("112", "빙그레요플레프로틴고단백질요거트");
            put("113", "산미구엘페일필젠캔");
            put("114", "산토리프리미엄몰트캔");
            put("115", "삼양패키징티즐피치우롱티");
            put("116", "삿포로프리미엄맥주캔");
            put("117", "서울우유");
            put("118", "서울우유스페셜티카페라떼마일드");
            put("119", "서울우유저지방");
            put("120", "서울우유초콜릿");
            put("121", "서울우유커피");
            put("122", "설화맥주캔");
            put("123", "셀렉스프로틴복숭아");
            put("124", "셀렉스프로틴아메리카노");
            put("125", "셀렉스프로틴초코");
            put("126", "소와나무덴마크드링킹요구르트베리믹스");
            put("127", "소와나무덴마크드링킹요구르트복숭아");
            put("128", "소화가잘되는우유미숫가루");
            put("129", "쉐퍼호퍼자몽맥주캔");
            put("130", "슈가로로레몬패트");
            put("131", "스마일리맥주캔");
            put("132", "스마일리몰디브맥주캔");
            put("133", "스타벅스더블샷바닐라");
            put("134", "스타벅스더블샷에스프레소와크림병");
            put("135", "스타벅스더블샷에스프레소와크림캔");
            put("136", "스타벅스브렉퍼스트블렌드블랙커피");
            put("137", "스타벅스스키니라떼200ml");
            put("138", "스타벅스스키니라떼270ml");
            put("139", "스타벅스시그니처초콜렛");
            put("140", "스타벅스오트에스프레소");
            put("141", "스타벅스카페라떼200ml");
            put("142", "스타벅스카페라떼270ml");
            put("143", "스타벅스카페라떼320ml");
            put("144", "스타벅스콜드브루돌체");
            put("145", "스타벅스콜드브루바닐라크림");
            put("146", "스타벅스파이크플레이스로스트블랙커피병");
            put("147", "스타벅스파이크플레이스로스트블랙커피캔");
            put("148", "스타벅스파이크플레이스로스트스위트블랙커피");
            put("149", "스타벅스프라푸치노돌체");
            put("150", "스타벅스프라푸치노모카");
            put("151", "스타벅스프라푸치노커피");
            put("152", "스텔라아르투아캔");
            put("153", "스트롱보우골든애플캔");
            put("154", "스트롱보우로제애플캔");
            put("155", "실론티페트500");
            put("156", "싱하캔");
            put("157", "써머스비사과캔");
            put("158", "썬업과일야채샐러드녹황");
            put("159", "썬업과일야채샐러드레드");
            put("160", "썬업과일야채샐러드퍼플");
            put("161", "썬업사과");
            put("162", "썬업오렌지");
            put("163", "썬키스트레몬에이드페트");
            put("164", "아몬드브리즈식이섬유");
            put("165", "아몬드브리즈오리지날");
            put("166", "아몬드브리즈초콜릿");
            put("167", "아몬드브리즈프로틴");
            put("168", "아사히캔");
            put("169", "아침에주스ABC210ml");
            put("170", "아침에주스사과210ml");
            put("171", "아카페라사이즈업카페라떼");
            put("172", "아크라거캔");
            put("173", "액티비아스무디골드키위사과");
            put("174", "액티비아스무디딸기바나나");
            put("175", "어메이징오트언스위트");
            put("176", "어메이징오트오리지날");
            put("177", "에딩거바이스비어캔");
            put("178", "오션스프레이피치크렌베리캔345ML");
            put("179", "옥토버훼스트바이젠캔");
            put("180", "요플레토핑다크초코");
            put("181", "요플레토핑쿠앤크");
            put("182", "웅진티즐유자그린티500ml");
            put("183", "이디야커피쇼콜라모카");
            put("184", "이디야커피카페라떼");
            put("185", "이디야커피토피넛시그니처라떼");
            put("186", "일화맥콜500ML");
            put("187", "일화맥콜250ML");
            put("188", "자연은토마토340ML");
            put("189", "장수생막걸리");
            put("190", "제이에게라거캔");
            put("191", "제이에게복숭아에일캔");
            put("192", "조지아맥스커피");
            put("193", "조지아오리지널");
            put("194", "조지아카페라떼");
            put("195", "좋은데이사과톡톡병");
            put("196", "좋은데이파인애플톡톡병");
            put("197", "진로소주병");
            put("198", "참이슬오리지널병");
            put("199", "참이슬후레쉬페트");
            put("200", "참이슬후레쉬병");
            put("201", "처음처럼페트");
            put("202", "처음처럼병");
            put("203", "처음처럼새로소주병");
            put("204", "천하장사에너지비어캔");
            put("205", "청하병");
            put("206", "칠성사이다355ML");
            put("207", "칠성사이다제로355ML");
            put("208", "칠성사이다캔250ML");
            put("209", "칠성사이다포도355ML");
            put("210", "칭따오논알콜릭");
            put("211", "칭따오위트비어캔");
            put("212", "칭따오캔");
            put("213", "카스0.0무알콜캔");
            put("214", "카스페트");
            put("215", "카스라이트캔");
            put("216", "카스캔");
            put("217", "카스프레시병");
            put("218", "카프리병");
            put("219", "카프리썬사파리");
            put("220", "카프리썬오렌지");
            put("221", "카프리썬오렌지망고");
            put("222", "칸타타라떼홀릭바닐라라떼");
            put("223", "칸타타스위트아메리카노병");
            put("224", "칸타타스위트아메리카노캔");
            put("225", "칸타타카라멜마끼아또병");
            put("226", "칸타타콜드브루블랙");
            put("227", "칸타타프리미엄라떼병");
            put("228", "칸타타프리미엄라떼캔");
            put("229", "칼스버그캔");
            put("230", "코어틴프로틴이온워터");
            put("231", "코젤다크캔");
            put("232", "코카스프라이트355ML");
            put("233", "코카스프라이트500ML");
            put("234", "코카스프라이트제로355ML");
            put("235", "코카콜라코카콜라350ML");
            put("236", "코카콜라250ML");
            put("237", "코카콜라제로250ML");
            put("238", "코카콜라제로500ML");
            put("239", "코카토레타500ML");
            put("240", "코카파워에이드MB240ML");
            put("241", "코카환타오렌지250ML");
            put("242", "코카환타오렌지600ML");
            put("243", "크로넨버그1664라거캔");
            put("244", "크로넨버그1664로제캔");
            put("245", "크로넨버그1664블랑캔");
            put("246", "클라우드생드래프트캔");
            put("247", "클라우드캔");
            put("248", "타이거라들러레몬캔");
            put("249", "타이거라들러자몽캔");
            put("250", "타이거캔");
            put("251", "테라페트");
            put("252", "테라캔");
            put("253", "테이크핏프로틴고소한맛");
            put("254", "테이크핏프로틴초코");
            put("255", "테일러푸룬주스");
            put("256", "트롤브루레몬캔");
            put("257", "트롤브루자몽캔");
            put("258", "티오피더블랙병");
            put("259", "티오피더블랙캔");
            put("260", "티오피마스터라떼병");
            put("261", "티오피마스터라떼캔");
            put("262", "티오피마일드에스프레소라떼");
            put("263", "티오피볼드에스프레소라떼");
            put("264", "티오피스모키라떼");
            put("265", "티오피스위트아메리카노병");
            put("266", "티오피스위트아메리카노캔");
            put("267", "티오피트루에스프레소블랙");
            put("268", "티오피트리플에스프레소라떼");
            put("269", "파스퇴르야채농장ABC");
            put("270", "파울라너뮌헨헬라거캔");
            put("271", "파울라너바이스비어캔");
            put("272", "펩시제로캔355ML");
            put("273", "펩시캔355ML");
            put("274", "풀무원액티비아복숭아");
            put("275", "필굿페트");
            put("276", "필굿엑스트라캔");
            put("277", "필굿캔");
            put("278", "필스너우르켈캔");
            put("279", "하이네켄실버캔");
            put("280", "하이네켄캔");
            put("281", "하이뮨프로틴밸런스액티브밀크");
            put("282", "하이뮨프로틴밸런스액티브초코");
            put("283", "하이트필라이트캔500ML");
            put("284", "하이트엑스트라콜드캔");
            put("285", "하이트제로무알콜캔");
            put("286", "하이트진로토닉워터300ML");
            put("287", "한맥캔");
            put("288", "해태아침에사과500ML");
            put("289", "해태갈배사이다500ML");
            put("290", "해태갈아만든배500ML");
            put("291", "해태썬키스트자몽소다350ML페트");
            put("292", "허쉬오리지널");
            put("293", "허쉬프로틴");
            put("294", "현대미에로화이바350ML");
            put("295", "호가든로제캔");
            put("296", "호가든보타닉캔");
            put("297", "호가든캔");
            put("298", "후디스키요그릭요거트젤리블루베리50그램");
            put("299", "후디스키요그릭요거트젤리청포도50그램");
            put("300", "후버사과주스");
        }};

        outData.rewind();
        outputMap.put(0, outData);
        Log.d("YoloV5Classifier", "mObjThresh: " + getObjThresh());

        Object[] inputArray = {imgData};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<Recognition>();

        float[][][] out = new float[1][output_box][numClass + 5];
        Log.d("YoloV5Classifier", "out[0] detect start");
        for (int i = 0; i < output_box; ++i) {
            for (int j = 0; j < numClass + 5; ++j) {
                if (isModelQuantized){
                    out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
                }
                else {
                    out[0][i][j] = byteBuffer.getFloat();
                }
            }
            // Denormalize xywh
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= getInputSize();
            }
        }

        // 각 bounding box에 대해 가장 확률이 높은 Class 예측
        for (int i = 0; i < output_box; ++i){
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            // 아래로 합침
//            for (int c = 0; c < labels.size(); ++c) {
//                classes[c] = out[0][i][5 + c];  // classes: 각 class의 확률 계산
//            }

            for (int c = 0; c < labels.size(); ++c) {
                classes[c] = out[0][i][5 + c];  // classes: 각 class의 확률 계산
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }   // 가장 큰 확률의 class로 선정
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];
                Log.d("YoloV5Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, nameMap.get(labels.get(detectedClass)),
                        confidenceInClass, rect, detectedClass));
            }
        }

        Log.d("YoloV5Classifier", "detect end");
        final ArrayList<Recognition> recognitions = nms(detections);
        return recognitions;
    }

    public boolean checkInvalidateBox(float x, float y, float width, float height, float oriW, float oriH, int intputSize) {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        float halfHeight = height / 2.0f;
        float halfWidth = width / 2.0f;

        float[] pred_coor = new float[]{x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight};

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        float resize_ratioW = 1.0f * intputSize / oriW;
        float resize_ratioH = 1.0f * intputSize / oriH;

        float resize_ratio = resize_ratioW > resize_ratioH ? resize_ratioH : resize_ratioW; //min

        float dw = (intputSize - resize_ratio * oriW) / 2;
        float dh = (intputSize - resize_ratio * oriH) / 2;

        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio;
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio;

        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio;
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio;

        // (3) clip some boxes those are out of range
        pred_coor[0] = pred_coor[0] > 0 ? pred_coor[0] : 0;
        pred_coor[1] = pred_coor[1] > 0 ? pred_coor[1] : 0;

        pred_coor[2] = pred_coor[2] < (oriW - 1) ? pred_coor[2] : (oriW - 1);
        pred_coor[3] = pred_coor[3] < (oriH - 1) ? pred_coor[3] : (oriH - 1);

        if ((pred_coor[0] > pred_coor[2]) || (pred_coor[1] > pred_coor[3])) {
            pred_coor[0] = 0;
            pred_coor[1] = 0;
            pred_coor[2] = 0;
            pred_coor[3] = 0;
        }

        // (4) discard some invalid boxes
        float temp1 = pred_coor[2] - pred_coor[0];
        float temp2 = pred_coor[3] - pred_coor[1];
        float temp = temp1 * temp2;
        if (temp < 0) {
            Log.e("checkInvalidateBox", "temp < 0");
            return false;
        }
        if (Math.sqrt(temp) > Float.MAX_VALUE) {
            Log.e("checkInvalidateBox", "temp max");
            return false;
        }

        return true;
    }
}
