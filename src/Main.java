import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {

    static double max(double data[][], int i) {
        double max = data[0][0];

        for (int j = 0; j < 130; j++)
            if (data[j][i] > max) max = data[j][i];


        return max;
    }

    static void add_collection(ArrayList<double[]> a[], double x[], int number ){

        double y[]=new double[x.length];
        for (int i=0;i<x.length;i++)
            y[i]=x[i];
        a[number].add(y);

    }


    public static void main(String[] args) {
        double buffer;
        FileWriter fileWriter,fileWriter_test;
        String string, s[];
        double max;
        double[] features=new double [16];


        double data[][]=new double [174][6];
        double test_data[][] = new double [29][6];
        Perceptrone_model perceptrone_neural_network = new Perceptrone_model(3,data[0].length,5,3);
        int l=0;
        try {
            Scanner scanner = new Scanner(new File("D:\\mlp_prod\\src\\new_train_data.txt"));
            Scanner scanner2  = new Scanner(new File("D:\\mlp_prod\\src\\new_test_data.txt"));
            while (scanner.hasNextLine()) {
                string = scanner.nextLine();
                s = string.split("\\,");
                for (int i = 0; i < 6; i++) {
                    buffer = Double.parseDouble(s[i]);
                    data[l][i]=buffer;
                }
                l++;
            }
            l=0;
            while (scanner2.hasNextLine()) {
                string = scanner2.nextLine();
                s = string.split("\\,");
                for (int i = 0; i < 6; i++) {
                    buffer = Double.parseDouble(s[i]);
                    test_data[l][i]=buffer;
                }
                l++;
            }
            perceptrone_neural_network.training(data,test_data);
            System.out.println("СКО = "+perceptrone_neural_network.erorr(data));
            perceptrone_neural_network.check_number_of_right(test_data);




               /* for (int j = data.length-5; j < data.length; j++) {
                    for (int k=1;k<14;k++) {
                        fileWriter_test.write(String.valueOf(data[j][k]));
                        fileWriter_test.write(" ");
                    }
                    fileWriter_test.write(String.valueOf(data[j][0]));
                    fileWriter_test.append(System.getProperty("line.separator"));
                }*/

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
