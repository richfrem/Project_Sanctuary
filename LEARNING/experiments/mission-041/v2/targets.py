import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import references
import mutations

def ref_lease(c):
    return references.lease(c)

def dev_lease(c):
    return None

def lease_unsigned_width_2(c):
    return mutations.evaluate('lease', 'unsigned_width_2', c)

def lease_identifier_casefold(c):
    return mutations.evaluate('lease', 'identifier_casefold', c)

def lease_stable_sequence_sort(c):
    return mutations.evaluate('lease', 'stable_sequence_sort', c)

def lease_sequence_tail_loss(c):
    return mutations.evaluate('lease', 'sequence_tail_loss', c)

def lease_signedness_erasure(c):
    return mutations.evaluate('lease', 'signedness_erasure', c)

def lease_identifier_prefix_encoding(c):
    return mutations.evaluate('lease', 'identifier_prefix_encoding', c)

def ref_renew(c):
    return references.renew(c)

def dev_renew(c):
    return None

def renew_unsigned_width_2(c):
    return mutations.evaluate('renew', 'unsigned_width_2', c)

def renew_identifier_casefold(c):
    return mutations.evaluate('renew', 'identifier_casefold', c)

def renew_stable_sequence_sort(c):
    return mutations.evaluate('renew', 'stable_sequence_sort', c)

def renew_sequence_tail_loss(c):
    return mutations.evaluate('renew', 'sequence_tail_loss', c)

def renew_signedness_erasure(c):
    return mutations.evaluate('renew', 'signedness_erasure', c)

def renew_identifier_prefix_encoding(c):
    return mutations.evaluate('renew', 'identifier_prefix_encoding', c)

def ref_release(c):
    return references.release(c)

def dev_release(c):
    return None

def release_unsigned_width_2(c):
    return mutations.evaluate('release', 'unsigned_width_2', c)

def release_identifier_casefold(c):
    return mutations.evaluate('release', 'identifier_casefold', c)

def release_stable_sequence_sort(c):
    return mutations.evaluate('release', 'stable_sequence_sort', c)

def release_sequence_tail_loss(c):
    return mutations.evaluate('release', 'sequence_tail_loss', c)

def release_signedness_erasure(c):
    return mutations.evaluate('release', 'signedness_erasure', c)

def release_identifier_prefix_encoding(c):
    return mutations.evaluate('release', 'identifier_prefix_encoding', c)

def ref_fence(c):
    return references.fence(c)

def dev_fence(c):
    return None

def fence_unsigned_width_2(c):
    return mutations.evaluate('fence', 'unsigned_width_2', c)

def fence_identifier_casefold(c):
    return mutations.evaluate('fence', 'identifier_casefold', c)

def fence_stable_sequence_sort(c):
    return mutations.evaluate('fence', 'stable_sequence_sort', c)

def fence_sequence_tail_loss(c):
    return mutations.evaluate('fence', 'sequence_tail_loss', c)

def fence_signedness_erasure(c):
    return mutations.evaluate('fence', 'signedness_erasure', c)

def fence_identifier_prefix_encoding(c):
    return mutations.evaluate('fence', 'identifier_prefix_encoding', c)

def ref_path(c):
    return references.path(c)

def dev_path(c):
    return None

def path_unsigned_width_2(c):
    return mutations.evaluate('path', 'unsigned_width_2', c)

def path_identifier_casefold(c):
    return mutations.evaluate('path', 'identifier_casefold', c)

def path_stable_sequence_sort(c):
    return mutations.evaluate('path', 'stable_sequence_sort', c)

def path_sequence_tail_loss(c):
    return mutations.evaluate('path', 'sequence_tail_loss', c)

def path_signedness_erasure(c):
    return mutations.evaluate('path', 'signedness_erasure', c)

def path_identifier_prefix_encoding(c):
    return mutations.evaluate('path', 'identifier_prefix_encoding', c)

def ref_relative(c):
    return references.relative(c)

def dev_relative(c):
    return None

def relative_unsigned_width_2(c):
    return mutations.evaluate('relative', 'unsigned_width_2', c)

def relative_identifier_casefold(c):
    return mutations.evaluate('relative', 'identifier_casefold', c)

def relative_stable_sequence_sort(c):
    return mutations.evaluate('relative', 'stable_sequence_sort', c)

def relative_sequence_tail_loss(c):
    return mutations.evaluate('relative', 'sequence_tail_loss', c)

def relative_signedness_erasure(c):
    return mutations.evaluate('relative', 'signedness_erasure', c)

def relative_identifier_prefix_encoding(c):
    return mutations.evaluate('relative', 'identifier_prefix_encoding', c)

def ref_extension(c):
    return references.extension(c)

def dev_extension(c):
    return None

def extension_unsigned_width_2(c):
    return mutations.evaluate('extension', 'unsigned_width_2', c)

def extension_identifier_casefold(c):
    return mutations.evaluate('extension', 'identifier_casefold', c)

def extension_stable_sequence_sort(c):
    return mutations.evaluate('extension', 'stable_sequence_sort', c)

def extension_sequence_tail_loss(c):
    return mutations.evaluate('extension', 'sequence_tail_loss', c)

def extension_signedness_erasure(c):
    return mutations.evaluate('extension', 'signedness_erasure', c)

def extension_identifier_prefix_encoding(c):
    return mutations.evaluate('extension', 'identifier_prefix_encoding', c)

def ref_path_key(c):
    return references.path_key(c)

def dev_path_key(c):
    return None

def path_key_unsigned_width_2(c):
    return mutations.evaluate('path_key', 'unsigned_width_2', c)

def path_key_identifier_casefold(c):
    return mutations.evaluate('path_key', 'identifier_casefold', c)

def path_key_stable_sequence_sort(c):
    return mutations.evaluate('path_key', 'stable_sequence_sort', c)

def path_key_sequence_tail_loss(c):
    return mutations.evaluate('path_key', 'sequence_tail_loss', c)

def path_key_signedness_erasure(c):
    return mutations.evaluate('path_key', 'signedness_erasure', c)

def path_key_identifier_prefix_encoding(c):
    return mutations.evaluate('path_key', 'identifier_prefix_encoding', c)

def ref_replay(c):
    return references.replay(c)

def dev_replay(c):
    return None

def replay_unsigned_width_2(c):
    return mutations.evaluate('replay', 'unsigned_width_2', c)

def replay_identifier_casefold(c):
    return mutations.evaluate('replay', 'identifier_casefold', c)

def replay_stable_sequence_sort(c):
    return mutations.evaluate('replay', 'stable_sequence_sort', c)

def replay_sequence_tail_loss(c):
    return mutations.evaluate('replay', 'sequence_tail_loss', c)

def replay_signedness_erasure(c):
    return mutations.evaluate('replay', 'signedness_erasure', c)

def replay_identifier_prefix_encoding(c):
    return mutations.evaluate('replay', 'identifier_prefix_encoding', c)

def ref_last_replay(c):
    return references.last_replay(c)

def dev_last_replay(c):
    return None

def last_replay_unsigned_width_2(c):
    return mutations.evaluate('last_replay', 'unsigned_width_2', c)

def last_replay_identifier_casefold(c):
    return mutations.evaluate('last_replay', 'identifier_casefold', c)

def last_replay_stable_sequence_sort(c):
    return mutations.evaluate('last_replay', 'stable_sequence_sort', c)

def last_replay_sequence_tail_loss(c):
    return mutations.evaluate('last_replay', 'sequence_tail_loss', c)

def last_replay_signedness_erasure(c):
    return mutations.evaluate('last_replay', 'signedness_erasure', c)

def last_replay_identifier_prefix_encoding(c):
    return mutations.evaluate('last_replay', 'identifier_prefix_encoding', c)

def ref_revision(c):
    return references.revision(c)

def dev_revision(c):
    return None

def revision_unsigned_width_2(c):
    return mutations.evaluate('revision', 'unsigned_width_2', c)

def revision_identifier_casefold(c):
    return mutations.evaluate('revision', 'identifier_casefold', c)

def revision_stable_sequence_sort(c):
    return mutations.evaluate('revision', 'stable_sequence_sort', c)

def revision_sequence_tail_loss(c):
    return mutations.evaluate('revision', 'sequence_tail_loss', c)

def revision_signedness_erasure(c):
    return mutations.evaluate('revision', 'signedness_erasure', c)

def revision_identifier_prefix_encoding(c):
    return mutations.evaluate('revision', 'identifier_prefix_encoding', c)

def ref_tombstone(c):
    return references.tombstone(c)

def dev_tombstone(c):
    return None

def tombstone_unsigned_width_2(c):
    return mutations.evaluate('tombstone', 'unsigned_width_2', c)

def tombstone_identifier_casefold(c):
    return mutations.evaluate('tombstone', 'identifier_casefold', c)

def tombstone_stable_sequence_sort(c):
    return mutations.evaluate('tombstone', 'stable_sequence_sort', c)

def tombstone_sequence_tail_loss(c):
    return mutations.evaluate('tombstone', 'sequence_tail_loss', c)

def tombstone_signedness_erasure(c):
    return mutations.evaluate('tombstone', 'signedness_erasure', c)

def tombstone_identifier_prefix_encoding(c):
    return mutations.evaluate('tombstone', 'identifier_prefix_encoding', c)

def ref_accounting(c):
    return references.accounting(c)

def dev_accounting(c):
    return None

def accounting_unsigned_width_2(c):
    return mutations.evaluate('accounting', 'unsigned_width_2', c)

def accounting_identifier_casefold(c):
    return mutations.evaluate('accounting', 'identifier_casefold', c)

def accounting_stable_sequence_sort(c):
    return mutations.evaluate('accounting', 'stable_sequence_sort', c)

def accounting_sequence_tail_loss(c):
    return mutations.evaluate('accounting', 'sequence_tail_loss', c)

def accounting_signedness_erasure(c):
    return mutations.evaluate('accounting', 'signedness_erasure', c)

def accounting_identifier_prefix_encoding(c):
    return mutations.evaluate('accounting', 'identifier_prefix_encoding', c)

def ref_clamp(c):
    return references.clamp(c)

def dev_clamp(c):
    return None

def clamp_unsigned_width_2(c):
    return mutations.evaluate('clamp', 'unsigned_width_2', c)

def clamp_identifier_casefold(c):
    return mutations.evaluate('clamp', 'identifier_casefold', c)

def clamp_stable_sequence_sort(c):
    return mutations.evaluate('clamp', 'stable_sequence_sort', c)

def clamp_sequence_tail_loss(c):
    return mutations.evaluate('clamp', 'sequence_tail_loss', c)

def clamp_signedness_erasure(c):
    return mutations.evaluate('clamp', 'signedness_erasure', c)

def clamp_identifier_prefix_encoding(c):
    return mutations.evaluate('clamp', 'identifier_prefix_encoding', c)

def ref_reserve(c):
    return references.reserve(c)

def dev_reserve(c):
    return None

def reserve_unsigned_width_2(c):
    return mutations.evaluate('reserve', 'unsigned_width_2', c)

def reserve_identifier_casefold(c):
    return mutations.evaluate('reserve', 'identifier_casefold', c)

def reserve_stable_sequence_sort(c):
    return mutations.evaluate('reserve', 'stable_sequence_sort', c)

def reserve_sequence_tail_loss(c):
    return mutations.evaluate('reserve', 'sequence_tail_loss', c)

def reserve_signedness_erasure(c):
    return mutations.evaluate('reserve', 'signedness_erasure', c)

def reserve_identifier_prefix_encoding(c):
    return mutations.evaluate('reserve', 'identifier_prefix_encoding', c)

def ref_refund(c):
    return references.refund(c)

def dev_refund(c):
    return None

def refund_unsigned_width_2(c):
    return mutations.evaluate('refund', 'unsigned_width_2', c)

def refund_identifier_casefold(c):
    return mutations.evaluate('refund', 'identifier_casefold', c)

def refund_stable_sequence_sort(c):
    return mutations.evaluate('refund', 'stable_sequence_sort', c)

def refund_sequence_tail_loss(c):
    return mutations.evaluate('refund', 'sequence_tail_loss', c)

def refund_signedness_erasure(c):
    return mutations.evaluate('refund', 'signedness_erasure', c)

def refund_identifier_prefix_encoding(c):
    return mutations.evaluate('refund', 'identifier_prefix_encoding', c)

def ref_schema(c):
    return references.schema(c)

def dev_schema(c):
    return None

def schema_unsigned_width_2(c):
    return mutations.evaluate('schema', 'unsigned_width_2', c)

def schema_identifier_casefold(c):
    return mutations.evaluate('schema', 'identifier_casefold', c)

def schema_stable_sequence_sort(c):
    return mutations.evaluate('schema', 'stable_sequence_sort', c)

def schema_sequence_tail_loss(c):
    return mutations.evaluate('schema', 'sequence_tail_loss', c)

def schema_signedness_erasure(c):
    return mutations.evaluate('schema', 'signedness_erasure', c)

def schema_identifier_prefix_encoding(c):
    return mutations.evaluate('schema', 'identifier_prefix_encoding', c)

def ref_migration(c):
    return references.migration(c)

def dev_migration(c):
    return None

def migration_unsigned_width_2(c):
    return mutations.evaluate('migration', 'unsigned_width_2', c)

def migration_identifier_casefold(c):
    return mutations.evaluate('migration', 'identifier_casefold', c)

def migration_stable_sequence_sort(c):
    return mutations.evaluate('migration', 'stable_sequence_sort', c)

def migration_sequence_tail_loss(c):
    return mutations.evaluate('migration', 'sequence_tail_loss', c)

def migration_signedness_erasure(c):
    return mutations.evaluate('migration', 'signedness_erasure', c)

def migration_identifier_prefix_encoding(c):
    return mutations.evaluate('migration', 'identifier_prefix_encoding', c)

def ref_required(c):
    return references.required(c)

def dev_required(c):
    return None

def required_unsigned_width_2(c):
    return mutations.evaluate('required', 'unsigned_width_2', c)

def required_identifier_casefold(c):
    return mutations.evaluate('required', 'identifier_casefold', c)

def required_stable_sequence_sort(c):
    return mutations.evaluate('required', 'stable_sequence_sort', c)

def required_sequence_tail_loss(c):
    return mutations.evaluate('required', 'sequence_tail_loss', c)

def required_signedness_erasure(c):
    return mutations.evaluate('required', 'signedness_erasure', c)

def required_identifier_prefix_encoding(c):
    return mutations.evaluate('required', 'identifier_prefix_encoding', c)

def ref_enum_map(c):
    return references.enum_map(c)

def dev_enum_map(c):
    return None

def enum_map_unsigned_width_2(c):
    return mutations.evaluate('enum_map', 'unsigned_width_2', c)

def enum_map_identifier_casefold(c):
    return mutations.evaluate('enum_map', 'identifier_casefold', c)

def enum_map_stable_sequence_sort(c):
    return mutations.evaluate('enum_map', 'stable_sequence_sort', c)

def enum_map_sequence_tail_loss(c):
    return mutations.evaluate('enum_map', 'sequence_tail_loss', c)

def enum_map_signedness_erasure(c):
    return mutations.evaluate('enum_map', 'signedness_erasure', c)

def enum_map_identifier_prefix_encoding(c):
    return mutations.evaluate('enum_map', 'identifier_prefix_encoding', c)
