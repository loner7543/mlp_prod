import com.sun.org.apache.xpath.internal.SourceTree;

import javax.xml.bind.SchemaOutputResolver;
import java.io.*;
import java.util.*;

public class Perceptrone_model {

    String cars[] = {"Audi","Bmw","Honda"};
    double weights[][][]; //веса сети; 1 компонента - номер ребер между i и i+1 cлоем; 2 компонента - i+i слой; 3 компонента - i-тый слой;
    double layers[][]; //значения нейронов; 1 - слой; 2 - номер нейрона;
    double output_of_layers[][]; //выходы нейронов; 1 - слой; 2 - номер нейрона;
    double biases[][]; //смещения; 1 - слой; 2 - номер нейрона;
    double errors[][]; //ошибки в нейронах; используется для алгоритма обратного распространения ошибки;
    double gradient_of_weights[][][]; //градиент для весов;
    int kol_of_layers; //количество слоев в нейронной сети;
    double gradient_of_biases[][]; //градиент для смещений;
    double learn_rate=2;//cкорость обучения;
    int mini_batch=1; //количество элементов для стохастичего градиента;
    int number_of_epoch=10000;
    double Lambda=0.48; //момент;
    double previouse_weights[][][];

    //конструктор; 1 параметр - количество слоев; 2 параметр - количество нейронов в каждом слое;
    public Perceptrone_model(int kol_layers, int... number_of_neurons_in_layer) {

        Random rand = new Random();

        if (kol_layers < 2) System.exit(0);
        layers = new double[kol_layers][];
        kol_of_layers = kol_layers;
        output_of_layers = new double[kol_layers][];
        biases = new double[kol_layers][];
        weights = new double[kol_layers - 1][][];
        errors = new double[kol_layers][];
        gradient_of_weights = new double[kol_layers - 1][][];
        gradient_of_biases = new double[kol_layers][];
        previouse_weights = new double [kol_layers-1][][];
        for (int i = 0; i < kol_layers; i++) {
            layers[i] = new double[number_of_neurons_in_layer[i]];
            output_of_layers[i] = new double[number_of_neurons_in_layer[i]];
            biases[i] = new double[number_of_neurons_in_layer[i]];
            gradient_of_biases[i] = new double[number_of_neurons_in_layer[i]];
            errors[i] = new double[number_of_neurons_in_layer[i]];
            for (int j = 0; j < biases[i].length; j++) {
                if (i != 0) biases[i][j] = rand.nextGaussian();
                else biases[i][j] = 0;
                gradient_of_biases[i][j]=0;
            }
            if (i != kol_layers - 1) {
                weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                previouse_weights[i]=new double[number_of_neurons_in_layer[i + 1]][];
                gradient_of_weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    previouse_weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    gradient_of_weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    for (int z = 0; z < number_of_neurons_in_layer[i]; z++) {
                        weights[i][j][z] = rand.nextGaussian();
                        gradient_of_weights[i][j][z]=0;
                        previouse_weights[i][j][z]=0;
                    }
                }
            }
        }
    }

    //инициализация весов 1 слоя;
    public void Set_the_first_layer(double[] x) {
        for (int i = 0; i < layers[0].length; i++) {
            layers[0][i] = x[i];
            output_of_layers[0][i] = x[i];
        }
    }

    //функция активации - сигмоида
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    //производная сигмоиды
    private double sigmoid_derivate(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    //прямое распротранение
    public void feed_forward() {
        double buffer = 0;

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < output_of_layers[i + 1].length; j++) {
                for (int z = 0; z < output_of_layers[i].length; z++)
                    buffer += weights[i][j][z] * output_of_layers[i][z];
                layers[i + 1][j] = buffer + biases[i + 1][j];
                output_of_layers[i + 1][j] = sigmoid(layers[i+1][j]);
                buffer=0;
            }
        }
    }

    //обнуление весов и смещений
    public void set_zero_parametres() {
        Random rand = new Random();

        for (int i = 0; i < kol_of_layers; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                if (i != 0) biases[i][j] = rand.nextGaussian();
                else biases[i][j] = 0;
                gradient_of_biases[i][j] = 0;
            }
            if (i != kol_of_layers - 1) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int z = 0; z < weights[i][j].length; z++) {
                        weights[i][j][z] = rand.nextGaussian();
                        gradient_of_weights[i][j][z] = 0;
                    }
                }
            }
        }
    }


    //обратное распространение ошибки
    public void back_forward(double[] y) {
        for (int i = 0; i < output_of_layers[kol_of_layers - 1].length; i++) {
            errors[kol_of_layers - 1][i] = (-y[i] + output_of_layers[kol_of_layers - 1][i]) * sigmoid_derivate(layers[kol_of_layers - 1][i]);
        }

        double buffer = 0;

        for (int i = kol_of_layers - 2; i >0; i--)
            for (int k = 0; k < output_of_layers[i].length; k++) {
                for (int z = 0; z < output_of_layers[i + 1].length; z++)
                    buffer += errors[i + 1][z] * weights[i][z][k];
                errors[i][k] = buffer * sigmoid_derivate(layers[i][k]);
                buffer=0;
            }

        for (int i=0;i<kol_of_layers-1;i++)
            for (int k=0;k<output_of_layers[i+1].length;k++)
                for (int z=0;z<output_of_layers[i].length;z++) {
                    gradient_of_weights[i][k][z]+=errors[i+1][k]*output_of_layers[i][z];
                }

        for (int i=1;i<kol_of_layers;i++)
            for (int j=0;j<biases[i].length;j++)
                gradient_of_biases[i][j]+=errors[i][j];
    }

    //изменение параметров сети
    public void change_weights(){
        for (int i=0;i<kol_of_layers-1;i++)
            for (int k=0;k<output_of_layers[i+1].length;k++)
                for (int z=0;z<output_of_layers[i].length;z++) {
                    weights[i][k][z]+= -gradient_of_weights[i][k][z]*learn_rate/(double)mini_batch+Lambda*(weights[i][k][z]-previouse_weights[i][k][z]);
                    gradient_of_weights[i][k][z]=0;
                    previouse_weights[i][k][z]=weights[i][k][z];
                }

        for (int i=1;i<kol_of_layers-1;i++)
            for (int j=0;j<biases[i].length;j++) {
                biases[i][j] -= gradient_of_biases[i][j] * learn_rate /(double) mini_batch;
                gradient_of_biases[i][j]=0;
            }
    }

    //перемешать обучающую выборку
    void mix_data(double data[][]){
        Random random = new Random();
        double buffer[];
        int num;

        for(int i=0;i<data.length;i++) {
            num = 0 + random.nextInt(data.length - 0);
            buffer = data[i];
            data[i] = data[num];
            data[num] = buffer;
        }

    }

    //ошибка для обучаемой выборки
    public double erorr(double data[][]) {

        double x[] = new double[layers[0].length+1];
        double error = 0;
        double errors[]=new double[data.length];
        for (int i=0;i<errors.length;i++) errors[i]=0;

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < layers[0].length; j++) {
                x[j] = data[i][j];
            }
            Set_the_first_layer(x);
            feed_forward();
            for (int j = 0; j < layers[kol_of_layers - 1].length; j++)
                if (j!= data[i][layers[0].length-1])
                    errors[i] += Math.pow(output_of_layers[kol_of_layers - 1][j] - 0, 2);
                else errors[i] += Math.pow(output_of_layers[kol_of_layers - 1][j] - 1, 2);
            errors[i]/=layers[kol_of_layers-1].length;
            error+=errors[i];
        }

        error/=data.length;

        error=Math.sqrt(error);

        return error;
    }

    //обучений нейронной сети
    public void training (double data[][],double test_data[][]  ){

        double accuracy=1;

        double y[]=new double [layers[kol_of_layers-1].length];
        double x[]=new double [layers[0].length];

        for (int l=0;l<number_of_epoch;l++) {
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < layers[kol_of_layers - 1].length; j++)
                    if (j != data[i][layers[0].length-1]) {
                        y[j] = 0;
                    }
                    else {
                        y[j] = 1;
                    }
                for (int j = 0; j < layers[0].length; j++)
                    x[j] = data[i][j];
                Set_the_first_layer(x);
                feed_forward();
                back_forward(y);
                if ((i+1)%mini_batch==0) change_weights();
                //System.out.println(erorr(test_data));
            }
            mix_data(data);
            // System.out.println(erorr(test_data));
            //  System.out.println(erorr(test_data));

        }

    }

    //установить количество нейронов в скрытом слое
    public void setHideLayer(int n){

        Random rand=new Random();

        int number_of_neurons_in_layer[]={layers[0].length,n,layers[layers.length-1].length};
        int kol_layers=kol_of_layers;
        layers = new double[kol_layers][];
        kol_of_layers = kol_layers;
        output_of_layers = new double[kol_layers][];
        biases = new double[kol_layers][];
        weights = new double[kol_layers - 1][][];
        errors = new double[kol_layers][];
        gradient_of_weights = new double[kol_layers - 1][][];
        gradient_of_biases = new double[kol_layers][];

        for (int i = 0; i < kol_layers; i++) {
            layers[i] = new double[number_of_neurons_in_layer[i]];
            output_of_layers[i] = new double[number_of_neurons_in_layer[i]];
            biases[i] = new double[number_of_neurons_in_layer[i]];
            gradient_of_biases[i] = new double[number_of_neurons_in_layer[i]];
            errors[i] = new double[number_of_neurons_in_layer[i]];
            for (int j = 0; j < biases[i].length; j++) {
                if (i != 0) biases[i][j] = rand.nextGaussian();
                else biases[i][j] = 0;
                gradient_of_biases[i][j]=0;
            }
            if (i != kol_layers - 1) {
                weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                gradient_of_weights[i] = new double[number_of_neurons_in_layer[i + 1]][];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    gradient_of_weights[i][j] = new double[number_of_neurons_in_layer[i]];
                    for (int z = 0; z < number_of_neurons_in_layer[i]; z++) {
                        weights[i][j][z] = rand.nextGaussian();
                        gradient_of_weights[i][j][z]=0;
                    }
                }
            }
        }

    }

    public void write_weights_to_file() {

        try {
            FileWriter fileWriter = new FileWriter("Preceptrone_weights.txt");
            for (int i = 0; i < kol_of_layers-1; i++) {
                for (int j = 0; j < output_of_layers[i+1].length; j++) {
                    for (int k = 0; k < output_of_layers[i].length; k++)
                        fileWriter.write(String.valueOf(weights[i][j]) + " ");
                    fileWriter.append(System.getProperty("line.separator"));
                }
                fileWriter.append(System.getProperty("line.separator"));
            }
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //нахождение нейрона с наибольшим выходом
    int find() {

        int max = 0;

        for (int i = 0; i < output_of_layers[kol_of_layers - 1].length; i++)
            if (output_of_layers[kol_of_layers - 1][max] < output_of_layers[kol_of_layers - 1][i])
                max = i;

        return max;
    }

    public void Setnumber_of_epoch(int n){
        number_of_epoch=n;
    }

    //проверка на правильность распознавания  выборки
    public double check_number_of_right(double data[][]){

        double x[]=new double[layers[0].length];
        int max;
        int kol=0;
        System.out.println("Тестовые данные      "+"    Ответ сети  ");
        for (int i=0;i<data.length;i++) {
            for (int j = 0; j < layers[0].length; j++)
                x[j] = data[i][j];
            Set_the_first_layer(x);
            feed_forward();
            max=find();
            System.out.println(cars[(int)data[i][layers[0].length-1]]+"                          "+cars[max]);
            if (max!=data[i][layers[0].length-1]) kol++;
        }
        System.out.println("Количество неправильно распознанных векторов = "+kol);

        return kol;
    }

    //получить выходной слой
    public double[] getOutput(){
        return output_of_layers[kol_of_layers-1];
    }

    //установить коэффициент обучения сети для построения графика
    public void setLearn_rate(double l){
        learn_rate=l;
    }
}
