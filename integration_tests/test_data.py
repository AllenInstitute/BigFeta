import os
from jinja2 import Environment, FileSystemLoader
import json


def render_json_template(env, template_file, **kwargs):
    template = env.get_template(template_file)
    d = json.loads(template.render(**kwargs))
    return d


TEST_DATA_ROOT = os.environ.get(
    'RENDER_TEST_DATA_ROOT',
    '/allen/aibs/pipeline/image_processing/volume_assembly')

test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
example_env = Environment(loader=FileSystemLoader(test_files_dir))

render_host = os.environ.get('RENDER_HOST', 'localhost')
render_port = os.environ.get('RENDER_PORT', 8080)
render_mongo_host = os.environ.get('RENDER_MONGO_HOST', 'localhost')
mongo_port = os.environ.get('MONGO_PORT', 27017)
render_owner = os.environ.get('RENDER_OWNER', "test")
render_test_owner = os.environ.get('RENDER_TEST_OWNER', 'test')
render_output_owner = os.environ.get("RENDER_OUTPUT_OWNER", "test")
client_script_location = os.environ.get(
        'RENDER_CLIENT_SCRIPTS',
        '/var/www/render/render-ws-java-client/src/main/scripts/')
project = 'test_project'
outdir = os.environ.get('BIGFETA_OUTPUT_DIR', '/home/danielk/tmp/tmp')

render_params = {
    'host': render_host,
    'port': render_port,
    'owner': 'test',
    'project': project,
    'client_scripts': client_script_location,
    'memGB': '1G',
    'validate_client': False
}

montage_raw_tilespecs_json = render_json_template(
        example_env,
        'montage_raw_tilespecs.json',
        test_data_root=TEST_DATA_ROOT)

montage_parameters = render_json_template(
        example_env, 'montage_test.json',
        render_project=project,
        render_host=render_host,
        render_owner=render_owner,
        render_port=render_port,
        render_mongo_host=render_mongo_host,
        mongo_port=mongo_port,
        render_output_owner=render_output_owner,
        render_client_scripts=client_script_location,
        solver_output_dir=outdir
        )

rough_output_stack = 'rough_result_similarity'
rough_parameters = render_json_template(
        example_env, 'rough_test.json',
        render_project=project,
        render_host=render_host,
        render_owner=render_owner,
        render_port=render_port,
        render_mongo_host=render_mongo_host,
        mongo_port=mongo_port,
        render_output_owner=render_output_owner,
        render_client_scripts=client_script_location,
        solver_output_dir=outdir,
        rough_output_stack=rough_output_stack
        )
