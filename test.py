# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize 
from wordcloud import WordCloud
from IPython.display import display, HTML
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# Extended File
from preprocessing import preprocess, preprocess2, extract_text_from_pdf, word_page_count, save_to_txt, remove_name_and_date
from clda import run_clda
from bert import run_bert
from hybrid_tf_idf import run_hybrid_tf_idf
from tf_idf import run_tf_idf
from rouge_metric import run_rouge_metric

def clda_runner(preprocessed_text_upload, preprocessed_text_pdf):
    # parameter CLDA
    start = 5
    end = 1
    iteration_num = 5
    clip = 50
    palpha = 0.05
    pgamma = 0.05
    pbeta = 0.05
    c_len = 10
    save_p = "Contextual_LDA_EXP" + str(clip) + "_" + str(c_len) + "_" + str(palpha) + "_" + str(pbeta) + "_" + str(pgamma) + "\\"
    
    for i in range(len(preprocessed_text_pdf)):
        st.subheader(f"Permodelan teks ke - {i + 1}")

        if len(preprocessed_text_upload) > 0:
            text = preprocess2(preprocessed_text_upload[i])
            upload_new = text.split(". ")
            for j in range(len(upload_new)):
                upload_new[j] = upload_new[j].split(" ")
        
        if len(preprocessed_text_upload) == 0:
            text = preprocess2(preprocessed_text_pdf[i])
            upload_new = text.split(". ")
            for k in range(len(upload_new)):
                upload_new[k] = upload_new[k].split(" ")
            upload_new = [item for item in upload_new if len(item) > 2]

        result = run_clda(upload_new, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
        if result is not None:
            st.dataframe(result[1])
            doc_lens = [len(d) for d in result[1].Text]

            # plot
            plt.figure(figsize=(10,5), dpi=160)
            plt.hist(doc_lens, bins=20, color='navy')
            #plt.text(50, 10009, "Mean   : " + str(round(np.mean(doc_lens))))
            #plt.text(50, 8295, "Median : " + str(round(np.median(doc_lens))))
            #plt.text(50, 6271, "Stdev   : " + str(round(np.std(doc_lens))))
            #plt.text(50, 4257, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
            #plt.text(50, 2225, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
            plt.title('Distribusi Jumlah Kalimat', fontdict=dict(size=20))
            
            plt.gca().set(xlim=(0, 100), ylabel='Jumlah Kalimat', xlabel='Panjang Kalimat')
            plt.tick_params(size=20)
            plt.xticks(np.linspace(0,100,10))
            
            st.pyplot(plt.gcf())
            

            # define subplot grid
            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(15, 12))
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle("Word Cloud Topik Dominan", fontsize=18, y=0.95)
                            
            df = result[1].copy()
            new_df = df.groupby(['dominant_topic'], as_index=False).agg({'sentence': ' '.join})

            # loop through tickers and axes
            for item, t, ax in zip(new_df.sentence, new_df.dominant_topic, axs.ravel()):
                word_cloud = WordCloud(collocations=False, background_color='white').generate(item)
                ax.imshow(word_cloud, interpolation='bilinear')
                ax.set_title(f"Topik - {t + 1}".upper())
                ax.set_xlabel("")
            st.pyplot(plt.gcf())
        else:
            st.error("Error")

def main():
    # Session State Handling
    if 'pdf_extracted' not in st.session_state:
        st.session_state.pdf_extracted = []

    if 'merged_pdf_text' not in st.session_state:
        st.session_state.merged_pdf_text = ''
    
    if 'extracted_file_showed' not in st.session_state:
        st.session_state.extracted_file_showed = 0
    
    if 'preprocessed_text' not in st.session_state:
        st.session_state.preprocessed_text = []

    if 'df_preprocessed' not in st.session_state:
        st.session_state.df_preprocessed = pd.DataFrame()

    if 'preprocessed_text_upload' not in st.session_state:
        st.session_state.preprocessed_text_upload = []

    if 'process_save_preprocess_file' not in st.session_state:
        st.session_state.process_save_preprocess_file = 0

    if 'preprocess_manual_upload_start' not in st.session_state:
        st.session_state.preprocess_manual_upload_start = 0

    if 'hasil_rouge_metric' not in st.session_state:
        st.session_state.hasil_rouge_metric = []

    if 'hasil_ringkas' not in st.session_state:
        st.session_state.hasil_ringkas = []

    if 'proses_ringkas' not in st.session_state:
        st.session_state.proses_ringkas = 0

    if 'file_ref' not in st.session_state:
        st.session_state.file_ref = []
    
    if 'butt_r' not in st.session_state:
        st.session_state.butt_r = 0

    if 'file_name_rouge' not in st.session_state:
        st.session_state.file_name_rouge = []


    # Main Page Design
    st.title(':bookmark_tabs: :blue[ALRISA]')
    st.header('_:blue[Aplikasi Peringkas Tesk Otomatis]_')
    st.write(
        """   
    Aplikasi ini merupakan instrumen yang dapat digunakan untuk mengekstraksi informasi dari file Dokumen Forum Penelaahan. Hasil ekstraksi berupa permodelan topik dan ringkasan dari file yang diunggah. Aplikasi dikembangkan dengan menggunakan model C-LDA dan Hybrid-TF-IDF.
             """
             )
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Tahapan Preprocessing:", ["- - - - -", 
                                                          "Ekstraksi", 
                                                          "Cari Kata", 
                                                          "Ekstraksi Tahap II"])
    
    menu2 = st.sidebar.selectbox("Peringkas Teks:", ["- - - - -", 
                                                     "Permodelan", 
                                                     "Ringkasan"])
    
    menu3 = st.sidebar.selectbox("Evaluasi Model:", ["- - - - -", 
                                                     "ROUGE Metric", 
                                                     "Perplexities"])
    

    # Menu Functions
    if menu == "Ekstraksi" and menu2 == "- - - - -" and menu3 == "- - - - -":
        st.subheader("Unggah file PDF untuk ekstraksi file")
        # define file uploader widget
        uploaded_files = st.file_uploader("Pilih file PDF",
                                          type=["pdf"],
                                          accept_multiple_files=True)
        
        button_extract = st.button("Ekstrak file")
        # click extract file button
        if button_extract:
            # check whether the uploaded files empty or not
            if len(uploaded_files) > 0:
                # get uploaded pdf texts
                for pdf_file in uploaded_files:
                    st.session_state.pdf_extracted.append(
                        extract_text_from_pdf(pdf_file)
                    )
                # success notification
                st.success("Ekstraksi PDF berhasil. File teks masih terpisah")
                # show extraction results
                st.subheader("Hasil ekstraksi")
                for idx, extracted_text in enumerate(st.session_state.pdf_extracted):
                    extracted_text = remove_name_and_date(extracted_text)
                    st.write(f"Hasil ekstraksi file {idx + 1}:")
                    st.text_area(f"File {idx + 1}", 
                                 value=extracted_text,
                                 height=300)
                    st.session_state.extracted_file_showed += 1
            else:
                # direction to upload pdf file first
                st.warning("File belum ter-upload. Upload terlebih dahulu")
            
        if st.session_state.extracted_file_showed == len(st.session_state.pdf_extracted) and len(st.session_state.pdf_extracted) > 0: # check all pdf files already shown    
    # join extraction    
            # join extraction
            button_join_extraction = st.button("Gabung Hasil Ekstraksi")
            if button_join_extraction:
            # Combine extracted texts into one
                merged_pdf_text = "\n".join([remove_name_and_date(text) for text in st.session_state.pdf_extracted])
                st.session_state.merged_pdf_text = merged_pdf_text
                st.success("Gabungan hasil ekstraksi teks berhasil!") # join success notification
                st.text_area("Gabungan hasil ekstraksi", value=st.session_state.merged_pdf_text, height=300)


    if menu == "Cari Kata" and menu2 == "- - - - -" and menu3 == "- - - - -":
        st.subheader("Mencari kata spesifik pada file yang diupload.")

        if len(st.session_state.pdf_extracted) > 0: # check pdf extracted is not empty
            search = st.text_input("Masukkan kata:")
            button_search_word = st.button("Cari Kata")
            if button_search_word:
                for idx, extracted_text in enumerate(st.session_state.pdf_extracted):
                    total, count = word_page_count(extracted_text, search)
                    st.write(f"Kata '{search}' ditemukan {total} kali pada {count} halaman pada dokumen ke - {idx+1}")
        else:
            st.warning("Silahkan unggah file pdf terlebih dahulu.")


    if menu == "Ekstraksi Tahap II" and menu2 == "- - - - -" and menu3 == "- - - - -":
        start_preprocess_button = st.button("Mulai Ekstraksi Tahap II")
        if start_preprocess_button:
            if len(st.session_state.pdf_extracted) > 0: # check pdf extracted is not empty
                if len(st.session_state.merged_pdf_text) == 0: # make sure the pdf text is not grouped or joined
                    for idx, extracted_text in enumerate(st.session_state.pdf_extracted):
                        extracted_text = remove_name_and_date(extracted_text)
                        preprocess_text = preprocess(extracted_text)
                        # print(preprocess_text)
                        st.session_state.preprocessed_text.append(preprocess_text) # preprocess extracted pdf text  

                if len(st.session_state.merged_pdf_text) > 0: # if pdf extracted is groupped or joined
                    preprocess_text = preprocess(st.session_state.merged_pdf_text)
                    st.session_state.preprocessed_text.append(preprocess_text)

        if len(st.session_state.preprocessed_text) > 0:
            st.subheader(f"Hasil Ekstraksi Tahap II")
            for idx, processed_text in enumerate(st.session_state.preprocessed_text):
                 st.text_area(f"Hasil Ekstraksi Tahap II - {idx + 1}", value=processed_text, height=300)
            st.subheader("Pilih bentuk penyimpanan Ekstraksi Tahap II file pdf")
            save_option = st.radio("", ["DataFrame", "File .txt"])
            if save_option == "File .txt":
                file_name = st.text_input("Masukkan nama file untuk disimpan (.txt):")
                if len(file_name) > 0 and st.button("Simpan"):
                    for j, preprocess_text in enumerate(st.session_state.preprocessed_text):
                        save_file_name = file_name + f" ke - {j + 1}"
                        save_to_txt(save_file_name, preprocess_text)
                        st.success(f"Sukses menyimpan Hasil Ekstraksi File '{save_file_name}'.")
                    st.success("Ekstraksi Tahap II file pdf telah selesai dan sudah disimpan dalam bentuk text!")
                    st.session_state.process_save_preprocess_file = 1
                elif len(file_name) == 0 and st.button("Simpan"):
                    st.warning("Masukkan nama file terlebih dahulu.")
 
            elif save_option == "DataFrame" and st.button("Simpan"):
                preprocessed_texts = [preprocess(text) for text in st.session_state.preprocessed_text]
                st.session_state.df_preprocessed = pd.DataFrame({"Preprocessed Text": preprocessed_texts})
                st.dataframe(st.session_state.df_preprocessed)
                st.success("Ekstraksi Tahap II file pdf telah selesai dan sudah disimpan dalam bentuk dataframe!")
                st.session_state.process_save_preprocess_file = 1

        if st.session_state.process_save_preprocess_file == 1:
            st.write("Ingin mengupload file preprocessing sendiri?")
            up_preprocess_file_button = st.button("Ya")
            not_up_preprocess_file_button = st.button("Tidak")

            if up_preprocess_file_button:
                st.session_state.preprocess_manual_upload_start = 1

            if not_up_preprocess_file_button:
                st.write("File preprocessing menggunakan hasil proses file pdf. Silahkan lanjutkan ke menu pemodelan.")
                st.session_state.process_save_preprocess_file = 0
                st.session_state.preprocess_manual_upload_start = 0
                st.session_state.preprocessed_text_upload = st.session_state.preprocessed_text

            if st.session_state.preprocess_manual_upload_start == 1:
                st.write("Unggah file preprocessing")
                uploaded_files1 = st.file_uploader("Pilih file txt", type=["txt"], accept_multiple_files=True)
                if len(uploaded_files1) > 0:
                    for uploaded_file in uploaded_files1:
                        bytes_data = uploaded_file.read()
                        bytes_data = bytes_data.decode()
                        st.session_state.preprocessed_text_upload.append(bytes_data)

                    st.success("Upload file preprocessing berhasil.")
                    st.write("File preprocessing yang diupload")
                    for idx, upload_preprocessing in enumerate(st.session_state.preprocessed_text_upload):
                        st.text_area(f"File preprocessing {idx + 1}" ,value=upload_preprocessing, height=300)


    if menu2 == "Permodelan" and menu == "- - - - -" and menu3 == "- - - - -":
        if len(st.session_state.preprocessed_text) > 0:
            st.subheader("Proses Preprocessing sudah Selesai. Mulai Pemodelan.")

            if len(st.session_state.preprocessed_text_upload) == 0:
                st.write("Proses Preprocessing Menggunakan Fungsi Preprocessing terhadap File PDF yang diupload.")
                # st.write(st.session_state.preprocessed_text)
                clda_runner(st.session_state.preprocessed_text_upload, st.session_state.preprocessed_text)
            else:
                st.write("Proses Preprocessing Menggunakan File txt yang diupload.")
                # st.write(st.session_state.preprocessed_text_upload)
                clda_runner(st.session_state.preprocessed_text_upload, st.session_state.preprocessed_text)
        else:
            st.subheader("Proses Preprocessing Belum dilakukan. Lakukan Preprocessing Terlebih Dahulu.")


    if menu2 == "Ringkasan" and menu == "- - - - -" and menu3 == "- - - - -":
        # Tampilkan pilihan untuk meringkas teks: BERT, TF-IDF, atau Hybrid-TF_IDF
        st.subheader("Pilih metode meringkas teks:")

        summary_option = st.radio("", ["BERT", "TF-IDF", "Hybrid-TF-IDF", "CLDA"])
        button_proses_ringkasan = st.button("Proses")
        # if button_proses_ringkasan:
        #     st.session_state.proses_ringkas = 1

        if summary_option == "CLDA":
            if button_proses_ringkasan:
                print("CLDA")
                st.session_state.hasil_ringkas = []

                # parameter CLDA
                start = 5
                end = 1
                iteration_num = 5
                clip = 50
                palpha = 0.05
                pgamma = 0.05
                pbeta = 0.05
                c_len = 10
                save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"

                for i in range(len(st.session_state.preprocessed_text)):
                    if len(st.session_state.preprocessed_text_upload) > 0:
                        upload_new = st.session_state.preprocessed_text_upload[i].split(". ")
                        for j in range(len(upload_new)):
                            upload_new[j] = upload_new[j].split(" ")
                    
                    if len(st.session_state.preprocessed_text_upload) == 0:
                        text = [" ".join(item) for item in st.session_state.preprocessed_text[i]][0]
                        upload_new = text.split(". ")
                        
                        for k in range(len(upload_new)):
                            upload_new[k] = upload_new[k].split(" ")
                        # upload_new = [item for item in upload_new if len(item) > 7]

                    result = run_clda(upload_new, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
                    
                    if result is not None:
                        st.session_state.hasil_ringkas.append(". ".join(result[0]))
                        st.text_area(f"Ringkasan CLDA {i + 1}" ,value=". ".join(result[0]), height=300)
                # st.session_state.proses_ringkas = 0
                st.session_state.hasil_rouge_metric = []

        if summary_option == "BERT":
            if button_proses_ringkasan:
                print("BERT")
                st.session_state.hasil_ringkas = []

                for i in range(len(st.session_state.preprocessed_text)):
                    if len(st.session_state.preprocessed_text_upload) > 0:
                        upload_new = st.session_state.preprocessed_text_upload[i]
                    
                    if len(st.session_state.preprocessed_text_upload) == 0:
                        text = [" ".join(item) for item in st.session_state.preprocessed_text[i]][0]
                        upload_new = text
                
                    result = run_bert(upload_new)
                    if result is not None:
                        st.session_state.hasil_ringkas.append(result)
                        st.text_area(f"Ringkasan BERT {i + 1}" ,value=result, height=300)
                st.session_state.proses_ringkas = 0
                st.session_state.hasil_rouge_metric = []

        if summary_option == "TF-IDF":
            if button_proses_ringkasan:
                print("TF-IDF")
                st.session_state.hasil_ringkas = []

                for i in range(len(st.session_state.preprocessed_text)):
                    if len(st.session_state.preprocessed_text_upload) > 0:
                        upload_new = st.session_state.preprocessed_text_upload[i]
                        
                    if len(st.session_state.preprocessed_text_upload) == 0:
                        text = [" ".join(item) for item in st.session_state.preprocessed_text[i]][0]
                        upload_new = text

                    result = run_tf_idf(upload_new)
                    if result is not None:
                        st.session_state.hasil_ringkas.append(result)
                        st.text_area(f"Ringkasan TF IDF {i + 1}" ,value=result, height=300)
                st.session_state.proses_ringkas = 0
                st.session_state.hasil_rouge_metric = []

        elif summary_option == "Hybrid-TF-IDF":
            if button_proses_ringkasan:
                print("Hybrid-TF-IDF")
                st.session_state.hasil_ringkas = []
                for i in range(len(st.session_state.preprocessed_text)):
                    if len(st.session_state.preprocessed_text_upload) > 0:
                        upload_new = st.session_state.preprocessed_text_upload[i]
                        
                    if len(st.session_state.preprocessed_text_upload) == 0:
                        text = [" ".join(item) for item in st.session_state.preprocessed_text[i]][0]
                        upload_new = text

                    result = run_hybrid_tf_idf(upload_new)
                    if result is not None:
                        st.session_state.hasil_ringkas.append(result)
                        st.text_area(f"Ringkasan Hybrid TF IDF {i + 1}" ,value=result, height=300)
                st.session_state.proses_ringkas = 0
                st.session_state.hasil_rouge_metric = []

    if menu3 == "ROUGE Metric" and menu == "- - - - -" and menu2 == "- - - - -":
        print("ROUGE Metric")
        st.write("Unggah file referensi.")

        uploaded_files1 = st.file_uploader("Pilih file txt", type=["txt"], accept_multiple_files=True)
        
        for uploaded_file in uploaded_files1:
            file_referensi = uploaded_file.read()
            file_referensi = file_referensi.decode()
            st.session_state.file_ref.append(file_referensi)
            st.session_state.file_name_rouge.append(uploaded_file.name)

        but_r = st.button("Tinjau Rouge Metric")

        if but_r:
            st.session_state.butt_r = 1

        if (len(st.session_state.hasil_ringkas)) > 0 and st.session_state.butt_r == 1:
            for idx, txtt in enumerate(st.session_state.hasil_ringkas):
                file_prediksi = txtt
                result_rouge_metric = run_rouge_metric(st.session_state.file_ref[idx], file_prediksi)
                st.session_state.hasil_rouge_metric.append(result_rouge_metric)
                p = pd.DataFrame(st.session_state.hasil_rouge_metric, 
                columns = ["Rouge-1 Precision", "Rouge-1 Recall", "Rouge-1 F-measure",
                             "Rouge-2 Precision", "Rouge-2 Recall", "Rouge-2 F-measure",
                             "Rouge-L Precision", "Rouge-L Recall", "Rouge-L F-measure"])
                
                p["Name"] = ''
                for i in range(len(p)):
                    p["Name"][i] = st.session_state.file_name_rouge[i]
                p = p[["Name", 
                "Rouge-1 Precision", "Rouge-1 Recall", "Rouge-1 F-measure",
               "Rouge-2 Precision", "Rouge-2 Recall", "Rouge-2 F-measure",
               "Rouge-L Precision", "Rouge-L Recall", "Rouge-L F-measure"]]

            st.dataframe(p)
            st.session_state.hasil_ringkas = []
            st.session_state.hasil_rouge_metric = []
            st.session_state.butt_r = 0
            st.session_state.file_ref = []
            st.session_state.file_name_rouge = []

    if menu3 == "Perplexities" and menu == "- - - - -" and menu2 == "- - - - -":
        print("Perplexities")
        start = 5
        end = 1
        iteration_num = 5
        clip = 50
        palpha = 0.05
        pgamma = 0.05
        pbeta = 0.05
        c_len = 10
        save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"

        for i in range(len(st.session_state.preprocessed_text)):
            if len(st.session_state.preprocessed_text_upload) > 0:
                upload_new = st.session_state.preprocessed_text_upload[i].split(". ")
                for j in range(len(upload_new)):
                    upload_new[j] = upload_new[j].split(" ")
        
            if len(st.session_state.preprocessed_text_upload) == 0:
                text = [" ".join(item) for item in st.session_state.preprocessed_text[i]][0]
                upload_new = text.split(". ")
                
                for k in range(len(upload_new)):
                    upload_new[k] = upload_new[k].split(" ")
                upload_new = [item for item in upload_new if len(item) > 2]

            st.write(f"Perplexities File {i + 1}:")
            dataset = save_p
            run_clda(upload_new, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
            y1 = np.load(str(dataset) +"C-LDAper_list"+ str(start) +".npy",allow_pickle=True)
            # print(y1)
            x = np.linspace(0, iteration_num+1, iteration_num+1)
            plt.figure(figsize=(8,5), dpi=160)
            plt.plot(x[::1], y1[0:], "r*-", label='C-LDA', linewidth=1)
            plt.title("Convergence Test By Perplexities")
            plt.ylabel(u"Perplexities")
            plt.xlabel(u"Iterations")
            plt.legend(loc="upper right")
            st.pyplot(plt.gcf())


if __name__=="__main__":
    main()
