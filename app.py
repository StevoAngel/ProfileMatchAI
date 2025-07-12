import os
import numpy as np
import pandas as pd
import streamlit as st
import utils.functions as functions

#from InstructorEmbedding import INSTRUCTOR


st.set_page_config(page_title="ProfileMatchAI", page_icon=":briefcase:", layout="wide")

################################ Load the files ###########################################
st.sidebar.title("Cargar archivos")
uploadedCvs = st.sidebar.file_uploader(
    "Arrastra los archivos (.pdf) de currículums aquí o haz clic para seleccionar",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploadedCvs:
    st.sidebar.success("Currículums cargado correctamente.")

uploadedJobDescription = st.sidebar.file_uploader(
    "Arrastra el archivo (.pdf) de la descripción del trabajo aquí o haz clic para seleccionar",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploadedJobDescription:
    st.sidebar.success("Descripción del trabajo cargada correctamente.")


def main():
    st.title("ProfileMatchAI: Match Your Profile with Job Descriptions")

    # Procesamiento
    if st.button("Procesar CVs"):
        if uploadedCvs and uploadedJobDescription:
            with st.spinner("Procesando CVs y descripción del puesto..."):
                # Procesa primero la descripción del puesto
                jobInfo = functions.extractJobDescriptionInfo(uploadedJobDescription)
                st.session_state["jobInfo"] = jobInfo

                # Inicializa la barra de progreso
                progress_bar = st.progress(0)
                cvsInfo = {}

                statusText = st.empty()

                total = len(uploadedCvs)
                for i, cv in enumerate(uploadedCvs):
                    # Procesa cada CV individualmente:
                    info = functions.extractSingleCVInfo(cv)
                    cvsInfo[info.name] = info  # Store the CV info in the dictionary
                    progress_bar.progress((i + 1) / total)
                    statusText.write(f"Procesando: {cv.name}")

                st.session_state["cvsInfo"] = cvsInfo
                st.success("Procesamiento completado.")
        else:
            st.warning("Debes cargar al menos un CV y la descripción del puesto.")

    #Create a DataFrame with the job description information
    if "jobInfo" in st.session_state:
        jobData = {key: value for key, value in st.session_state["jobInfo"].model_dump().items()}
        dfJobDesc = pd.DataFrame([jobData])
        dfJobDesc.columns = [
            "Titulo", "Descripción", "Responsabilidades",
            "Habilidades Técnicas", "Habilidades Blandas", "Ubicación"
        ]
        st.subheader("📄 Información de la Descripción de Puesto")
        st.dataframe(dfJobDesc)

    # Create a DataFrame with the CV information
    if "cvsInfo" in st.session_state:
        cvData = {key: value.model_dump() for key, value in st.session_state["cvsInfo"].items()} # Convertir a diccionario
        dfCVs = pd.DataFrame(cvData).T # Convertir a DataFrame
        dfCVs.columns = ["Nombre", "Correo", "Teléfono", "Perfil", "Ubicación", "Experiencia", "Educación", "Habilidades Técnicas", "Habilidades Blandas"] # Renombrar columnas
        #dfCVs.set_index("Nombre", inplace=True) # Establecer la columna "name" como índice
        dfCVs = dfCVs[~dfCVs.index.duplicated(keep='first')] # Eliminar duplicados

        st.subheader("📄 Información de los candidatos")
        st.dataframe(dfCVs)

        #Option to download the CVs information as an Excel file
        excelFile = functions.df2Excel(dfCVs)
        st.download_button(
            label="📥 Descargar en Excel",
            data=excelFile,
            file_name="info_candidatos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()