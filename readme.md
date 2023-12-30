AI/ML projects explored for upskilling in MLOps and ML Engineering. 

## Deployment steps (GCP)
1. Dockerize each project and build corresponding image
2. Push docker image to Google's Artifact Registry. Google Container Registry (GCR) is to be decommissioned soon so might as well learn their recommended image registry.

   i.e.
   - `docker build . -t qamodel`
   - `docker tag qamodel us-docker.pkg.dev/earnest-trilogy-409716/testrepository/qamodel`
   - `gcloud auth configure-docker us-docker.pkg.dev`
   - `gcloud auth login`
   - `docker push us-docker.pkg.dev/earnest-trilogy-409716/testrepository/qamodel`

   where 
   - `us-docker.pkg.dev` is the region of the artifact repository,
   - `us-docker.pkg.dev/earnest-trilogy-409716/testrepository` is the repository

   ![img.png](img.png)
3. Deploy using either Google Cloud run or through a dedicated VM
4. Test API using accessible external IP or UI.